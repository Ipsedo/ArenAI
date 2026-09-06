//
// Created by samuel on 06/09/2026.
//

#include "./liquid_ppo_trainer.h"

#include <algorithm>
#include <fstream>

#include "../../distributions/multinomial.h"
#include "../../distributions/truncated_normal.h"
#include "../../metrics/mean_metric.h"
#include "../../networks/constants.h"
#include "../../networks_utils/print_module.h"
#include "../../networks_utils/torch_saver.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {
    namespace {
        // merges the [T, nb_tanks] leading dimensions into a single row dimension
        torch::Tensor flatten_steps(const torch::Tensor &tensor) {
            auto sizes = tensor.sizes().vec();
            sizes.erase(sizes.begin());
            sizes[0] = tensor.size(0) * tensor.size(1);
            return tensor.reshape(sizes);
        }

        constexpr float LOG_RATIO_MAX_ABS = 3.f;

        constexpr float KL_TRIM_FRACTION = 0.01f;

        constexpr float CONTINUOUS_ALPHA_K_P = 2e-1f;
        constexpr float CONTINUOUS_ALPHA_K_I = 5e-3f;
        constexpr float CONTINUOUS_ALPHA_K_D = 1.f;

        constexpr float DISCRETE_ALPHA_K_P = 2e-1f;
        constexpr float DISCRETE_ALPHA_K_I = 1e-2f;
        constexpr float DISCRETE_ALPHA_K_D = 1.f;

        constexpr float ALPHA_INITIAL = 1e-3f;

    }// namespace

    LiquidPpoTrainer::LiquidPpoTrainer(
        const std::shared_ptr<LiquidActor> &actor,
        const std::shared_ptr<LiquidPpoRolloutBuffer> &rollout_buffer, const int vision_height,
        const int vision_width, const int nb_sensors, const int nb_continuous_actions,
        const int nb_discrete_action, const float actor_learning_rate,
        const float critic_learning_rate, const int hidden_size_sensors,
        const std::vector<std::tuple<int, int>> &vision_channels,
        const std::vector<int> &group_norm_nums, const int neuron_number, const int unfolding_steps,
        const float delta_t, const torch::Device device, const int metric_window_size,
        const float gamma, const float gae_lambda, const float clip_epsilon, const float target_kl,
        const float grad_norm_max, const float continuous_target_entropy,
        const float discrete_target_entropy_factor, const int epochs, const int rollout_size,
        const int minibatch_size, const int chunk_size)
        : actor(actor), rollout_buffer(rollout_buffer),
          continuous_alpha(std::make_unique<PidLagrangianAlphaParameters>(
              CONTINUOUS_ALPHA_K_P, CONTINUOUS_ALPHA_K_I, CONTINUOUS_ALPHA_K_D, ALPHA_INITIAL,
              nb_continuous_actions)),
          discrete_alpha(std::make_unique<PidLagrangianAlphaParameters>(
              DISCRETE_ALPHA_K_P, DISCRETE_ALPHA_K_I, DISCRETE_ALPHA_K_D, ALPHA_INITIAL, 1)),
          continuous_target_entropy(continuous_target_entropy),
          discrete_target_entropy(
              discrete_target_entropy_factor * multinomial_maximum_entropy(nb_discrete_action)),
          critic(std::make_shared<LiquidCritic>(
              vision_height, vision_width, nb_sensors, hidden_size_sensors, vision_channels,
              group_norm_nums, neuron_number, unfolding_steps, delta_t)),
          actor_optim(
              std::make_unique<torch::optim::Adam>(this->actor->parameters(), actor_learning_rate)),
          critic_optim(
              std::make_unique<torch::optim::Adam>(critic->parameters(), critic_learning_rate)),
          actor_mean_loss_metric(std::make_shared<MeanMetric>("π", metric_window_size)),
          critic_mean_loss_metric(std::make_shared<MeanMetric>("v", metric_window_size)),
          explained_variance_metric(std::make_shared<MeanMetric>("ev", metric_window_size)),
          continuous_entropy_metric(std::make_shared<MeanMetric>("Hc", metric_window_size)),
          discrete_entropy_metric(std::make_shared<MeanMetric>("Hd", metric_window_size)),
          continuous_alpha_metric(std::make_shared<MeanMetric>("α_c", metric_window_size, 2, true)),
          discrete_alpha_metric(std::make_shared<MeanMetric>("α_d", metric_window_size, 2, true)),
          clip_fraction_metric(std::make_shared<MeanMetric>("clip", metric_window_size)),
          kl_metric(std::make_shared<MeanMetric>("kl", metric_window_size, 2, true)),
          skip_fraction_metric(std::make_shared<MeanMetric>("skip", metric_window_size)),
          gamma(gamma), gae_lambda(gae_lambda), clip_epsilon(clip_epsilon), target_kl(target_kl),
          grad_norm_max(grad_norm_max), epochs(epochs), rollout_size(rollout_size),
          minibatch_size(minibatch_size), chunk_size(chunk_size) {
        to(device);

        set_train(false);
    }

    void LiquidPpoTrainer::step() {
        if (rollout_buffer->nb_complete_steps() >= static_cast<size_t>(rollout_size)) train();
    }

    void LiquidPpoTrainer::train() {
        const auto device = actor->parameters().back().device();

        const auto rollout = rollout_buffer->get_rollout();

        set_train(false);
        const auto [advantages, returns, critic_hiddens] = compute_gae(rollout, device);

        set_train(true);

        const auto nb_steps = rollout.rewards.size(0);
        const auto nb_tanks = rollout.rewards.size(1);

        // contiguous chunk_size-long windows per tank; the (at most chunk_size - 1)
        // trailing steps that cannot form a full chunk are dropped
        const auto nb_chunks_per_tank = nb_steps / chunk_size;
        if (nb_chunks_per_tank == 0) return;
        const auto nb_kept_steps = nb_chunks_per_tank * chunk_size;

        // the rollout stays on CPU as flat [T * nb_tanks, ...] views; only the
        // minibatches hit the device
        const auto flat = [&](const torch::Tensor &tensor) {
            return flatten_steps(tensor.slice(0, 0, nb_kept_steps));
        };

        const auto flat_vision = flat(rollout.states.vision);
        const auto flat_proprioception = flat(rollout.states.proprioception);
        const auto flat_continuous_actions = flat(rollout.actions.continuous_action);
        const auto flat_discrete_actions = flat(rollout.actions.discrete_action);
        const auto flat_old_log_probs =
            flat(rollout.continuous_log_probs) + flat(rollout.discrete_log_probs);
        const auto flat_advantages = flat(advantages);
        const auto flat_returns = flat(returns);
        const auto flat_valids = flat(rollout.valids);
        const auto flat_actor_hiddens = flat(rollout.actor_hiddens);
        const auto flat_critic_hiddens = flat(critic_hiddens);

        // chunk unit u = chunk * nb_tanks + tank; the ones without any live
        // transition bring nothing to train on
        const auto chunk_valid_counts = rollout.valids.slice(0, 0, nb_kept_steps)
                                            .reshape({nb_chunks_per_tank, chunk_size, nb_tanks})
                                            .sum(1)
                                            .flatten();
        const auto valid_chunk_idx = torch::nonzero(chunk_valid_counts > 0).squeeze(-1);
        const auto nb_valid_chunks = valid_chunk_idx.size(0);
        if (nb_valid_chunks == 0) return;

        const auto chunks_per_minibatch = std::max<int64_t>(1, minibatch_size / chunk_size);
        const auto step_offsets = torch::arange(chunk_size);

        for (int e = 0; e < epochs; e++) {
            const auto perm = valid_chunk_idx.index_select(0, torch::randperm(nb_valid_chunks));

            for (int64_t start = 0; start < nb_valid_chunks; start += chunks_per_minibatch) {
                const auto idx = perm.slice(
                    0, start, std::min<int64_t>(start + chunks_per_minibatch, nb_valid_chunks));

                const auto tank = idx.remainder(nb_tanks);
                const auto start_step = idx.div(nb_tanks, "floor") * chunk_size;

                // flat rows of the minibatch's chunks and of their first steps
                const auto rows = ((start_step.unsqueeze(1) + step_offsets.unsqueeze(0)) * nb_tanks
                                   + tank.unsqueeze(1))
                                      .flatten();
                const auto start_rows = start_step * nb_tanks + tank;

                // [nb_chunks, chunk_size, ...]
                const auto select_chunks = [&](const torch::Tensor &flat_tensor) {
                    auto sizes = flat_tensor.sizes().vec();
                    sizes[0] = idx.size(0);
                    sizes.insert(sizes.begin() + 1, chunk_size);
                    return flat_tensor.index_select(0, rows).reshape(sizes).to(device);
                };
                // [nb_chunks, neuron_number]
                const auto select_hiddens = [&](const torch::Tensor &flat_hiddens) {
                    return flat_hiddens.index_select(0, start_rows).to(device);
                };

                const auto mb_vision = select_chunks(flat_vision);
                const auto mb_proprioception = select_chunks(flat_proprioception);
                const auto mb_valids = select_chunks(flat_valids);

                train_actor(
                    mb_vision, mb_proprioception, select_hiddens(flat_actor_hiddens),
                    select_chunks(flat_continuous_actions), select_chunks(flat_discrete_actions),
                    select_chunks(flat_old_log_probs), select_chunks(flat_advantages), mb_valids);

                train_critic(
                    mb_vision, mb_proprioception, select_hiddens(flat_critic_hiddens),
                    select_chunks(flat_returns), mb_valids);
            }
        }

        set_train(false);
    }

    bool LiquidPpoTrainer::train_actor(
        const torch::Tensor &vision, const torch::Tensor &proprioception,
        const torch::Tensor &hidden, const torch::Tensor &continuous_actions,
        const torch::Tensor &discrete_actions, const torch::Tensor &old_log_probs,
        const torch::Tensor &advantages, const torch::Tensor &valids) const {
        const auto device = actor->parameters().back().device();

        const auto out = actor->act_sequence(vision, proprioception, hidden);

        // recurrence done: fold time into rows and keep the live transitions only
        const auto valid_idx = torch::nonzero(valids.flatten(0, 1).squeeze(-1)).squeeze(-1);
        const auto rows = [&](const torch::Tensor &tensor) {
            return tensor.flatten(0, 1).index_select(0, valid_idx);
        };

        const auto mu = rows(out.mu);
        const auto sigma = rows(out.sigma);
        const auto discrete_proba = rows(out.discrete);

        const auto curr_continuous_log_probs =
            truncated_normal_log_pdf(rows(continuous_actions), mu, sigma).sum(-1, true);

        const auto clamped_proba = torch::clamp(discrete_proba, EPSILON, 1.0 - EPSILON);
        const auto curr_discrete_log_probs =
            torch::sum(rows(discrete_actions) * torch::log(clamped_proba), -1, true);

        const auto log_ratio = torch::clamp(
            curr_continuous_log_probs + curr_discrete_log_probs - rows(old_log_probs),
            -LOG_RATIO_MAX_ABS, LOG_RATIO_MAX_ABS);

        const auto ratio = torch::exp(log_ratio);

        const auto continuous_entropy = truncated_normal_entropy(mu, sigma);
        const auto discrete_entropy = multinomial_entropy(discrete_proba);

        const auto kl_per_row = (ratio - 1.f - log_ratio).flatten();

        const auto nb_kept = std::max<int64_t>(
            1, static_cast<int64_t>(
                   static_cast<float>(kl_per_row.size(0)) * (1.f - KL_TRIM_FRACTION)));

        // per-row KL is non-negative: sorting ascending puts the outliers past nb_kept
        const auto approx_kl =
            std::get<0>(torch::sort(kl_per_row)).slice(0, 0, nb_kept).mean().item<float>();

        const bool kl_exceeded = target_kl > 0.f && approx_kl > 1.5f * target_kl;

        const auto entropy_bonus =
            torch::sum(continuous_alpha->alpha().detach() * continuous_entropy, -1)
            + discrete_alpha->alpha().squeeze(1).detach() * discrete_entropy;

        if (!kl_exceeded) {
            const auto clipped_ratio = torch::clamp(ratio, 1.f - clip_epsilon, 1.f + clip_epsilon);
            const auto valid_advantages = rows(advantages);
            const auto surrogate =
                torch::min(ratio * valid_advantages, clipped_ratio * valid_advantages);

            const auto actor_loss = -torch::mean(surrogate + entropy_bonus);

            actor_optim->zero_grad();
            actor_loss.backward();
            torch::nn::utils::clip_grad_norm_(actor->parameters(), grad_norm_max);
            actor_optim->step();

            // actor metrics
            actor_mean_loss_metric->add(actor_loss.cpu().item<float>());
        }

        // adjust alphas
        continuous_alpha->update(
            continuous_entropy, torch::tensor(continuous_target_entropy, device));
        discrete_alpha->update(discrete_entropy, torch::tensor(discrete_target_entropy, device));

        // metrics
        continuous_entropy_metric->add(continuous_entropy.mean().item<float>());
        discrete_entropy_metric->add(discrete_entropy.mean().item<float>());

        continuous_alpha_metric->add(continuous_alpha->alpha().mean().item<float>());
        discrete_alpha_metric->add(discrete_alpha->alpha().mean().item<float>());

        kl_metric->add(approx_kl);

        clip_fraction_metric->add(
            ((ratio - 1.f).abs() > clip_epsilon).to(torch::kFloat).mean().item<float>());

        skip_fraction_metric->add(kl_exceeded ? 1.f : 0.f);

        return kl_exceeded;
    }

    void LiquidPpoTrainer::train_critic(
        const torch::Tensor &vision, const torch::Tensor &proprioception,
        const torch::Tensor &hidden, const torch::Tensor &returns,
        const torch::Tensor &valids) const {
        const auto out = critic->value_sequence(vision, proprioception, hidden);

        // recurrence done: fold time into rows and keep the live transitions only
        const auto valid_idx = torch::nonzero(valids.flatten(0, 1).squeeze(-1)).squeeze(-1);
        const auto values = out.value.flatten(0, 1).index_select(0, valid_idx);
        const auto valid_returns = returns.flatten(0, 1).index_select(0, valid_idx);

        const auto critic_loss = torch::mse_loss(values, valid_returns, at::Reduction::Mean);

        critic_optim->zero_grad();
        critic_loss.backward();
        torch::nn::utils::clip_grad_norm_(critic->parameters(), grad_norm_max);
        critic_optim->step();

        critic_mean_loss_metric->add(critic_loss.cpu().item<float>());

        const auto residual_var = (valid_returns - values.detach()).var(false);
        const auto returns_var = valid_returns.var(false);
        explained_variance_metric->add(
            (1.f - residual_var / returns_var.clamp_min(EPSILON)).item<float>());
    }

    LiquidGaeResult
    LiquidPpoTrainer::compute_gae(const LiquidPpoRollout &rollout, const torch::Device device) {
        torch::NoGradGuard no_grad;

        const auto nb_steps = rollout.rewards.size(0);
        const auto nb_tanks = rollout.rewards.size(1);

        const auto episode_starts = rollout.episode_starts.accessor<bool, 1>();

        if (!critic_carry.defined() || critic_carry.size(0) != nb_tanks)
            critic_carry = critic->initial_state(static_cast<int>(nb_tanks));

        // sequential critic pass: the values and the liquid state opening each step
        std::vector<torch::Tensor> value_list;
        std::vector<torch::Tensor> hidden_list;
        value_list.reserve(nb_steps);
        hidden_list.reserve(nb_steps);

        for (int64_t t = 0; t < nb_steps; t++) {
            if (episode_starts[t]) critic_carry = critic->initial_state(static_cast<int>(nb_tanks));

            hidden_list.push_back(critic_carry.cpu());

            const auto out = critic->value(
                rollout.states.vision[t].to(device), rollout.states.proprioception[t].to(device),
                critic_carry);

            value_list.push_back(out.value.cpu());
            critic_carry = out.next_x;
        }

        const auto values = torch::stack(value_list, 0);
        const auto critic_hiddens = torch::stack(hidden_list, 0);

        // the carry is NOT advanced past the bootstrap: its observation opens the
        // next rollout's first step
        const auto bootstrap_value =
            critic
                ->value(
                    rollout.bootstrap_state.vision.to(device),
                    rollout.bootstrap_state.proprioception.to(device), critic_carry)
                .value.cpu()
                .unsqueeze(0);

        // next values are the values shifted by one step, closed by the bootstrap state
        const auto next_values = torch::cat({values.slice(0, 1), bootstrap_value}, 0);

        const auto rewards = rollout.rewards.to(torch::kFloat);
        const auto dones = rollout.dones.to(torch::kFloat);
        const auto valids = rollout.valids.to(torch::kFloat);

        const auto deltas = rewards + gamma * next_values * (1.f - dones) - values;

        auto advantages = torch::zeros_like(deltas);
        auto gae = torch::zeros({nb_tanks, 1}, deltas.options());
        for (int64_t t = nb_steps - 1; t >= 0; t--) {
            gae = deltas[t] + gamma * gae_lambda * (1.f - dones[t]) * gae;
            advantages[t] = gae;
        }

        const auto returns = advantages + values;

        const auto nb_valid = valids.sum().clamp_min(1.f);
        const auto advantage_mean = torch::sum(advantages * valids) / nb_valid;
        const auto advantage_std =
            torch::sqrt(torch::sum(torch::square(advantages - advantage_mean) * valids) / nb_valid);
        advantages = (advantages - advantage_mean) / (advantage_std + EPSILON);

        return {.advantages = advantages, .returns = returns, .critic_hiddens = critic_hiddens};
    }

    std::vector<std::shared_ptr<AbstractMetric>> LiquidPpoTrainer::get_metrics() {
        return {actor_mean_loss_metric,    critic_mean_loss_metric, explained_variance_metric,
                continuous_entropy_metric, continuous_alpha_metric, discrete_entropy_metric,
                discrete_alpha_metric,     clip_fraction_metric,    kl_metric,
                skip_fraction_metric};
    }

    void LiquidPpoTrainer::save(const std::filesystem::path &output_folder) {
        // Models
        save_torch(output_folder, actor, "actor.pt");
        save_torch(output_folder, critic, "critic.pt");

        // Optimizers
        save_torch(output_folder, actor_optim, "actor_optim.pt");
        save_torch(output_folder, critic_optim, "critic_optim.pt");

        // string repr
        std::ostringstream actor_repr_oss;
        dump_module_tree(actor, actor_repr_oss, 0, "actor");
        std::ofstream actor_repr_file(output_folder / "actor_repr.txt");
        actor_repr_file << actor_repr_oss.str();
        actor_repr_file.close();

        std::ostringstream critic_repr_oss;
        dump_module_tree(critic, critic_repr_oss, 0, "critic");
        std::ofstream critic_repr_file(output_folder / "critic_repr.txt");
        critic_repr_file << critic_repr_oss.str();
        critic_repr_file.close();
    }

    void LiquidPpoTrainer::set_train(const bool train) const {
        actor->train(train);
        critic->train(train);

        continuous_alpha->train(train);
        discrete_alpha->train(train);
    }

    void LiquidPpoTrainer::to(const torch::Device device) const {
        actor->to(device);
        critic->to(device);

        continuous_alpha->to(device);
        discrete_alpha->to(device);
    }

    int LiquidPpoTrainer::count_parameters() {
        return count_parameters_impl(actor->parameters())
               + count_parameters_impl(critic->parameters());
    }
}// namespace arenai::agent
