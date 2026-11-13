#include "./ppg_trainer.h"

#include <algorithm>
#include <fstream>
#include <numeric>

#include "../../distributions/bernoulli.h"
#include "../../distributions/gaussian_tanh.h"
#include "../../metrics/mean_metric.h"
#include "../../networks/constants.h"
#include "../../networks_utils/print_module.h"
#include "../../networks_utils/torch_saver.h"
#include "../ppo_objective.h"
#include "../update_utils.h"

namespace arenai::agent {
    namespace {
        // merges the [T, nb_tanks] leading dimensions into a single row dimension
        torch::Tensor flatten_steps(const torch::Tensor &tensor) {
            auto sizes = tensor.sizes().vec();
            sizes.erase(sizes.begin());
            sizes[0] = tensor.size(0) * tensor.size(1);
            return tensor.reshape(sizes);
        }

        int64_t total_rows(const std::vector<torch::Tensor> &blocks) {
            return std::accumulate(
                blocks.begin(), blocks.end(), int64_t{0},
                [](const int64_t sum, const torch::Tensor &idx) { return sum + idx.size(0); });
        }

        void write_module_repr(
            const std::filesystem::path &output_folder,
            const std::shared_ptr<torch::nn::Module> &module, const std::string &name) {
            std::ostringstream oss;
            dump_module_tree(module, oss, 0, name);
            std::ofstream file(output_folder / (name + "_repr.txt"));
            file << oss.str();
        }

    }// namespace

    PpgTrainer::PpgTrainer(
        const std::shared_ptr<DistillationActor> &actor,
        const std::shared_ptr<PpgRolloutBuffer> &rollout_buffer, const int nb_continuous_actions,
        const int nb_discrete_actions, const int nb_sensors, const float actor_learning_rate,
        const float critic_learning_rate, const int hidden_size_sensors,
        const std::vector<int> &critic_hidden_sizes,
        const std::vector<std::tuple<int, int>> &vision_channels,
        const std::vector<int> &group_norm_nums, const torch::Device device,
        const int metric_window_size, const float gamma, const float gae_lambda,
        const float aim_gae_lambda, const float clip_epsilon, const float target_kl,
        const float grad_norm_max, const float initial_alpha, const float alpha_learning_rate,
        const std::vector<float> &continuous_target_entropy, const float discrete_target_proba,
        const int actor_epochs, const int critic_epochs, const int rollout_size,
        const int actor_batch_size, const int actor_gpu_batch_size, const int critic_batch_size,
        const int critic_gpu_batch_size, const int actor_aux_epochs, const int critic_aux_epochs,
        const float beta_clone, const int n_pi)
        : actor(actor), rollout_buffer(rollout_buffer),
          alphas(std::make_unique<EntropyAlphas>(
              nb_continuous_actions, nb_discrete_actions, initial_alpha, alpha_learning_rate,
              continuous_target_entropy, discrete_target_proba)),
          critic(std::make_shared<ValueFunction>(
              nb_sensors, hidden_size_sensors, critic_hidden_sizes, vision_channels,
              group_norm_nums)),
          actor_optim(
              std::make_unique<torch::optim::Adam>(this->actor->parameters(), actor_learning_rate)),
          critic_optim(
              std::make_unique<torch::optim::Adam>(critic->parameters(), critic_learning_rate)),
          return_scaler(std::make_shared<ReturnScaler>(gamma)),
          actor_mean_loss_metric(std::make_shared<MeanMetric>("π", metric_window_size)),
          critic_mean_loss_metric(std::make_shared<MeanMetric>("v", metric_window_size)),
          explained_variance_metric(std::make_shared<MeanMetric>("ev", metric_window_size)),
          continuous_entropy_metric(std::make_shared<MeanMetric>("Hc", metric_window_size)),
          discrete_entropy_metric(std::make_shared<MeanMetric>("Hd", metric_window_size)),
          clip_fraction_metric(std::make_shared<MeanMetric>("clip", metric_window_size)),
          kl_metric(std::make_shared<MeanMetric>("kl", metric_window_size, 2, true)),
          skip_fraction_metric(std::make_shared<MeanMetric>("skip", metric_window_size)),
          skip_canon_metric(std::make_shared<MeanMetric>("skip_canon", metric_window_size)),
          gamma(gamma), gae_lambda(gae_lambda), aim_gae_lambda(aim_gae_lambda),
          clip_epsilon(clip_epsilon), target_kl(target_kl), grad_norm_max(grad_norm_max),
          actor_epochs(actor_epochs), critic_epochs(critic_epochs), rollout_size(rollout_size),
          actor_batch_size(actor_batch_size), actor_gpu_batch_size(actor_gpu_batch_size),
          critic_batch_size(critic_batch_size), critic_gpu_batch_size(critic_gpu_batch_size),
          actor_aux_epochs(actor_aux_epochs), critic_aux_epochs(critic_aux_epochs),
          beta_clone(beta_clone), n_pi(std::max(1, n_pi)) {
        to(device);

        set_train(false);
    }

    void PpgTrainer::step() {
        if (rollout_buffer->nb_complete_steps() < static_cast<size_t>(rollout_size)) return;

        nb_policy_phases++;
        train(nb_policy_phases % n_pi == 0);
    }

    PpgTrainer::FlatRows PpgTrainer::FlatRows::select(const torch::Tensor &idx) const {
        const auto rows = [&](const torch::Tensor &tensor) { return tensor.index_select(0, idx); };
        return {
            .vision = rows(vision),
            .proprioception = rows(proprioception),
            .continuous_pre_tanh = rows(continuous_pre_tanh),
            .discrete_actions = rows(discrete_actions),
            .fire_masks = rows(fire_masks),
            .old_continuous_log_probs = rows(old_continuous_log_probs),
            .old_discrete_log_probs = rows(old_discrete_log_probs),
            .advantages = rows(advantages),
            .aim_advantages = rows(aim_advantages),
            .returns = rows(returns)};
    }

    PpgTrainer::CloneTargets PpgTrainer::CloneTargets::select(const torch::Tensor &idx) const {
        return {
            .mus = mus.index_select(0, idx),
            .sigmas = sigmas.index_select(0, idx),
            .discrete_probas = discrete_probas.index_select(0, idx)};
    }

    void PpgTrainer::train(const bool run_aux) const {
        const auto device = actor->parameters().back().device();

        auto rollout = rollout_buffer->get_rollout();

        rollout.states = rollout.states.to(device);

        set_train(false);
        const auto [advantages, aim_advantages, returns] = compute_gae(rollout, device);

        // flat [T * nb_tanks, ...] rows, all on the device
        const auto flat = [&](const torch::Tensor &tensor) {
            return flatten_steps(tensor).to(device);
        };
        const FlatRows rows = {
            .vision = flat(rollout.states.get_vision()),
            .proprioception = flat(rollout.states.get_state()),
            .continuous_pre_tanh = flat(rollout.continuous_pre_tanhs),
            .discrete_actions = flat(rollout.actions.discrete_action),
            .fire_masks = flat(rollout.states.get_fire_mask()),
            .old_continuous_log_probs = flat(rollout.continuous_log_probs),
            .old_discrete_log_probs = flat(rollout.discrete_log_probs),
            .advantages = flat(advantages),
            .aim_advantages = flat(aim_advantages),
            .returns = flat(returns)};

        // live (step, tank) pairs; the optimizer batches split these rows only
        const auto valid_idx =
            torch::nonzero(flatten_steps(rollout.valids).squeeze(-1)).squeeze(-1).to(device);
        if (valid_idx.size(0) == 0) return;

        set_train(true);
        run_actor_epochs(rows, valid_idx);
        run_critic_epochs(critic_epochs, rows, valid_idx);

        if (run_aux) auxiliary_phase(rows, valid_idx);

        set_train(false);
    }

    void PpgTrainer::run_actor_epochs(const FlatRows &rows, const torch::Tensor &valid_idx) const {
        GroupKlStop kl_stop(target_kl);

        for (int e = 0; e < actor_epochs; e++) {
            const auto batches =
                shuffled_batches(valid_idx, actor_batch_size, actor_gpu_batch_size);
            kl_stop.on_planned_steps(static_cast<int>(batches.size()));

            for (const auto &blocks: batches) {
                if (!kl_stop.any_active()) break;

                const auto kls = accumulate_actor_gradients(rows, blocks, kl_stop.active());
                kl_metric->add(kls.joint);

                if (kl_stop.stop_exceeded(kls)) {
                    if (!kl_stop.any_active()) break;
                    accumulate_actor_gradients(rows, blocks, kl_stop.active());
                }

                torch::nn::utils::clip_grad_norm_(actor->parameters(), grad_norm_max);
                actor_optim->step();
                alphas->step();
                kl_stop.on_step();
            }
        }

        skip_fraction_metric->add(kl_stop.rest_skip_fraction());
        skip_canon_metric->add(kl_stop.aim_skip_fraction());
    }

    GroupKls PpgTrainer::accumulate_actor_gradients(
        const FlatRows &rows, const std::vector<torch::Tensor> &blocks,
        const ActiveGroups &active) const {
        actor_optim->zero_grad();
        alphas->zero_grad();

        const auto nb_batch_rows = static_cast<float>(total_rows(blocks));

        std::vector<RowKls> row_kls;
        for (const auto &idx: blocks)
            row_kls.push_back(backward_actor(
                rows.select(idx), static_cast<float>(idx.size(0)) / nb_batch_rows, active));

        return trimmed_group_kls(row_kls);
    }

    void PpgTrainer::run_critic_epochs(
        const int nb_epochs, const FlatRows &rows, const torch::Tensor &valid_idx) const {
        for (int e = 0; e < nb_epochs; e++)
            for (const auto &blocks:
                 shuffled_batches(valid_idx, critic_batch_size, critic_gpu_batch_size)) {
                critic_optim->zero_grad();

                const auto nb_batch_rows = static_cast<float>(total_rows(blocks));

                for (const auto &idx: blocks)
                    backward_critic(
                        rows.select(idx), static_cast<float>(idx.size(0)) / nb_batch_rows);

                torch::nn::utils::clip_grad_norm_(critic->parameters(), grad_norm_max);
                critic_optim->step();
            }
    }

    void PpgTrainer::auxiliary_phase(const FlatRows &rows, const torch::Tensor &valid_idx) const {
        set_train(false);
        const auto targets = compute_clone_targets(rows);
        set_train(true);

        run_actor_aux_epochs(rows, targets, valid_idx);
        run_critic_epochs(critic_aux_epochs, rows, valid_idx);
    }

    PpgTrainer::CloneTargets PpgTrainer::compute_clone_targets(const FlatRows &rows) const {
        torch::NoGradGuard no_grad;

        std::vector<torch::Tensor> mu_chunks;
        std::vector<torch::Tensor> sigma_chunks;
        std::vector<torch::Tensor> discrete_chunks;

        for (int64_t i = 0; i < rows.vision.size(0); i += critic_gpu_batch_size) {
            const auto end = std::min<int64_t>(i + critic_gpu_batch_size, rows.vision.size(0));

            const auto [mu, sigma, discrete_proba] =
                actor->act(rows.vision.slice(0, i, end), rows.proprioception.slice(0, i, end));
            mu_chunks.push_back(mu);
            sigma_chunks.push_back(sigma);
            discrete_chunks.push_back(discrete_proba);
        }

        return {
            .mus = torch::cat(mu_chunks, 0),
            .sigmas = torch::cat(sigma_chunks, 0),
            .discrete_probas = torch::cat(discrete_chunks, 0)};
    }

    void PpgTrainer::run_actor_aux_epochs(
        const FlatRows &rows, const CloneTargets &targets, const torch::Tensor &valid_idx) const {
        // value distillation in the actor, its policy held by the clone term
        for (int e = 0; e < actor_aux_epochs; e++) {
            for (const auto &blocks:
                 shuffled_batches(valid_idx, actor_batch_size, actor_gpu_batch_size)) {
                actor_optim->zero_grad();

                const auto nb_batch_rows = static_cast<float>(total_rows(blocks));

                for (const auto &idx: blocks)
                    backward_actor_aux(
                        rows.select(idx), targets.select(idx),
                        static_cast<float>(idx.size(0)) / nb_batch_rows);

                torch::nn::utils::clip_grad_norm_(actor->parameters(), grad_norm_max);
                actor_optim->step();
            }
        }
    }

    RowKls PpgTrainer::backward_actor(
        const FlatRows &block, const float batch_fraction, const ActiveGroups &active) const {
        const auto [mu, sigma, discrete_proba] = actor->act(block.vision, block.proprioception);
        const auto discrete_mask =
            TorchState::discrete_action_mask(block.fire_masks, discrete_proba.size(-1));

        const auto current = PolicyLogProbs::split(
            gaussian_tanh_log_pdf(block.continuous_pre_tanh, mu, sigma),
            (bernoulli_log_proba(block.discrete_actions, discrete_proba) * discrete_mask)
                .sum(-1, true));
        const auto old =
            PolicyLogProbs::split(block.old_continuous_log_probs, block.old_discrete_log_probs);

        const auto [surrogate, log_ratio, ratio, kls] = clipped_surrogate(
            current, old, block.aim_advantages, block.advantages, clip_epsilon, active);

        const auto continuous_entropy = gaussian_tanh_entropy(mu, sigma);
        const auto discrete_entropy = bernoulli_entropy(discrete_proba) * discrete_mask;

        // the alphas are constants for the actor: they learn from their own dual loss
        const auto entropy_bonus =
            (alphas->continuous().detach() * continuous_entropy
             * continuous_group_mask(active, continuous_entropy.size(-1), mu.device()))
                .sum(-1, true)
            + static_cast<float>(active.rest)
                  * (alphas->discrete().detach() * discrete_entropy).sum(-1, true);

        const auto actor_loss = -torch::mean(surrogate + entropy_bonus);

        // weighted so that the accumulated gradient is the one of the whole rollout's mean
        (actor_loss * batch_fraction).backward();
        alphas->backward(continuous_entropy, discrete_entropy, discrete_mask, batch_fraction);

        // metrics
        actor_mean_loss_metric->add(actor_loss.cpu().item<float>());
        continuous_entropy_metric->add(continuous_entropy.mean().item<float>());
        // mean over the discrete actions that had an effect
        discrete_entropy_metric->add(
            (discrete_entropy.sum() / discrete_mask.sum().clamp_min(1.f)).item<float>());

        clip_fraction_metric->add(
            ((ratio - 1.f).abs() > clip_epsilon).to(torch::kFloat).mean().item<float>());

        return kls;
    }

    void PpgTrainer::backward_actor_aux(
        const FlatRows &block, const CloneTargets &targets, const float batch_fraction) const {
        const auto values = actor->value(block.vision, block.proprioception);
        const auto [mu, sigma, discrete_proba] = actor->act(block.vision, block.proprioception);

        const auto aux_loss = torch::mse_loss(values, block.returns, at::Reduction::None);

        // KL(old || new), closed form for both: the tanh squash leaves the normal's KL unchanged
        const auto continuous_kl =
            gaussian_tanh_kl(targets.mus, targets.sigmas, mu, sigma).sum(-1, true);

        const auto old_p = targets.discrete_probas.clamp(EPSILON, 1.f - EPSILON);
        const auto new_p = discrete_proba.clamp(EPSILON, 1.f - EPSILON);
        // a masked fire is not a decision: kept out of the clone term as of the policy phase
        const auto discrete_kl =
            ((old_p * torch::log(old_p / new_p)
              + (1.f - old_p) * torch::log((1.f - old_p) / (1.f - new_p)))
             * TorchState::discrete_action_mask(block.fire_masks, discrete_proba.size(-1)))
                .sum(-1, true);

        const auto loss = torch::mean(aux_loss + beta_clone * (continuous_kl + discrete_kl));

        // weighted so that the accumulated gradient is the one of the whole batch's mean
        (loss * batch_fraction).backward();
    }

    void PpgTrainer::backward_critic(const FlatRows &block, const float batch_fraction) const {
        const auto values = critic->value(block.vision, block.proprioception);
        const auto critic_loss = torch::mse_loss(values, block.returns, at::Reduction::Mean);

        (critic_loss * batch_fraction).backward();

        critic_mean_loss_metric->add(critic_loss.cpu().item<float>());

        const auto residual_var = (block.returns - values.detach()).var(false);
        const auto returns_var = block.returns.var(false);
        explained_variance_metric->add(
            (1.f - residual_var / returns_var.clamp_min(EPSILON)).item<float>());
    }

    PpgGaeResult
    PpgTrainer::compute_gae(const PpgRollout &rollout, const torch::Device device) const {
        torch::NoGradGuard no_grad;

        const auto nb_steps = rollout.rewards.size(0);
        const auto nb_tanks = rollout.rewards.size(1);

        // critic forward in device-sized chunks: the whole rollout's activations do not fit
        const auto eval_values = [&](const torch::Tensor &vision,
                                     const torch::Tensor &proprioception) {
            std::vector<torch::Tensor> chunks;
            for (int64_t i = 0; i < vision.size(0); i += critic_gpu_batch_size) {
                const auto end = std::min<int64_t>(i + critic_gpu_batch_size, vision.size(0));
                chunks.push_back(critic
                                     ->value(
                                         vision.slice(0, i, end).to(device),
                                         proprioception.slice(0, i, end).to(device))
                                     .cpu());
            }
            return torch::cat(chunks, 0);
        };

        const auto values = eval_values(
                                flatten_steps(rollout.states.get_vision()),
                                flatten_steps(rollout.states.get_state()))
                                .reshape({nb_steps, nb_tanks, 1});

        // next values are the values shifted by one step, closed by the bootstrap state
        const auto bootstrap_value =
            eval_values(rollout.bootstrap_state.get_vision(), rollout.bootstrap_state.get_state())
                .unsqueeze(0);
        const auto next_values = torch::cat({values.slice(0, 1), bootstrap_value}, 0);

        const auto rewards = return_scaler->scale(
            rollout.rewards, rollout.dones.logical_or(rollout.truncateds), rollout.valids);
        const auto dones = rollout.dones.to(torch::kFloat);
        const auto truncateds = rollout.truncateds.to(torch::kFloat);
        const auto valids = rollout.valids.to(torch::kFloat);

        const auto cuts = torch::clamp_max(dones + truncateds, 1.f);
        const auto deltas = rewards + gamma * next_values * (1.f - cuts) - values;

        const auto advantages = generalized_advantages(deltas, cuts, gamma * gae_lambda);

        return {
            .advantages = normalized(advantages, valids),
            .aim_advantages =
                aim_advantages(rollout.aim_rewards, cuts, valids, gamma, aim_gae_lambda),
            .returns = advantages + values};
    }

    std::vector<std::shared_ptr<AbstractMetric>> PpgTrainer::get_metrics() {
        return {
            actor_mean_loss_metric,
            critic_mean_loss_metric,
            explained_variance_metric,
            continuous_entropy_metric,
            discrete_entropy_metric,
            clip_fraction_metric,
            kl_metric,
            skip_fraction_metric};
    }

    std::vector<std::shared_ptr<AbstractMetric>> PpgTrainer::get_csv_only_metrics() {
        auto metrics = alphas->get_metrics();
        metrics.push_back(skip_canon_metric);
        return metrics;
    }

    void PpgTrainer::save(const std::filesystem::path &output_folder) {
        // Models
        save_torch(output_folder, actor, "actor.pt");
        save_torch(output_folder, critic, "critic.pt");

        // Optimizers
        save_torch(output_folder, actor_optim, "actor_optim.pt");
        save_torch(output_folder, critic_optim, "critic_optim.pt");
        save_torch(output_folder, return_scaler, "return_scaler.pt");
        alphas->save(output_folder);

        // string repr
        write_module_repr(output_folder, actor, "actor");
        write_module_repr(output_folder, critic, "critic");
    }

    void PpgTrainer::set_train(const bool train) const {
        actor->train(train);
        critic->train(train);
    }

    void PpgTrainer::to(const torch::Device device) const {
        actor->to(device);
        critic->to(device);
        alphas->to(device);
    }

    int PpgTrainer::count_parameters() {
        return count_parameters_impl(actor->parameters())
               + count_parameters_impl(critic->parameters()) + alphas->count_parameters();
    }
}// namespace arenai::agent
