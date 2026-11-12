#include "./liquid_ppo_trainer.h"

#include <algorithm>
#include <fstream>
#include <numeric>

#include "../../distributions/bernoulli.h"
#include "../../distributions/gaussian_tanh.h"
#include "../../metrics/mean_metric.h"
#include "../../networks/constants.h"
#include "../../networks_utils/print_module.h"
#include "../../networks_utils/torch_loader.h"
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

        void write_module_repr(
            const std::filesystem::path &output_folder,
            const std::shared_ptr<torch::nn::Module> &module, const std::string &name) {
            std::ostringstream oss;
            dump_module_tree(module, oss, 0, name);
            std::ofstream file(output_folder / (name + "_repr.txt"));
            file << oss.str();
        }

    }// namespace

    LiquidPpoTrainer::LiquidPpoTrainer(
        const std::shared_ptr<LiquidActor> &actor,
        const std::shared_ptr<LiquidPpoRolloutBuffer> &rollout_buffer,
        const int nb_continuous_actions, const int nb_discrete_actions, const int nb_sensors,
        const float actor_learning_rate, const float critic_learning_rate,
        const int hidden_size_sensors, const std::vector<std::tuple<int, int>> &vision_channels,
        const std::vector<int> &group_norm_nums, const int neuron_number, const int unfolding_steps,
        const float delta_t, const torch::Device device, const int metric_window_size,
        const float gamma, const float gae_lambda, const float aim_gae_lambda,
        const float clip_epsilon, const float target_kl, const float grad_norm_max,
        const float initial_alpha, const float alpha_learning_rate,
        const std::vector<float> &continuous_target_entropy, const float discrete_target_proba,
        const int actor_epochs, const int critic_epochs, const int rollout_size,
        const int actor_batch_size, const int actor_gpu_batch_size, const int critic_batch_size,
        const int critic_gpu_batch_size, const int chunk_size)
        : actor(actor), rollout_buffer(rollout_buffer),
          alphas(std::make_unique<EntropyAlphas>(
              nb_continuous_actions, nb_discrete_actions, initial_alpha, alpha_learning_rate,
              continuous_target_entropy, discrete_target_proba)),
          critic(std::make_shared<LiquidCritic>(
              nb_sensors, hidden_size_sensors, vision_channels, group_norm_nums, neuron_number,
              unfolding_steps, delta_t)),
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
          gamma(gamma), gae_lambda(gae_lambda), aim_gae_lambda(aim_gae_lambda),
          clip_epsilon(clip_epsilon), target_kl(target_kl), grad_norm_max(grad_norm_max),
          actor_epochs(actor_epochs), critic_epochs(critic_epochs), rollout_size(rollout_size),
          actor_batch_size(actor_batch_size), actor_gpu_batch_size(actor_gpu_batch_size),
          critic_batch_size(critic_batch_size), critic_gpu_batch_size(critic_gpu_batch_size),
          chunk_size(chunk_size) {
        to(device);

        set_train(false);
    }

    void LiquidPpoTrainer::step() {
        if (rollout_buffer->nb_complete_steps() >= static_cast<size_t>(rollout_size)) train();
    }

    LiquidPpoTrainer::FlatRows LiquidPpoTrainer::FlatRows::select_chunks(
        const torch::Tensor &rows, const torch::Tensor &start_rows,
        const int64_t chunk_size) const {
        const auto chunked = [&](const torch::Tensor &tensor) {
            auto sizes = tensor.sizes().vec();
            sizes[0] = rows.size(0) / chunk_size;
            sizes.insert(sizes.begin() + 1, chunk_size);
            return tensor.index_select(0, rows).reshape(sizes);
        };
        return {
            .vision = chunked(vision),
            .proprioception = chunked(proprioception),
            .continuous_pre_tanh = chunked(continuous_pre_tanh),
            .discrete_actions = chunked(discrete_actions),
            .fire_masks = chunked(fire_masks),
            .old_continuous_log_probs = chunked(old_continuous_log_probs),
            .old_discrete_log_probs = chunked(old_discrete_log_probs),
            .advantages = chunked(advantages),
            .aim_advantages = chunked(aim_advantages),
            .returns = chunked(returns),
            .valids = chunked(valids),
            .actor_hiddens = actor_hiddens.index_select(0, start_rows),
            .critic_hiddens = critic_hiddens.index_select(0, start_rows)};
    }

    LiquidPpoTrainer::ChunkIndex::ChunkIndex(
        const torch::Tensor &valids, const int64_t nb_tanks, const int64_t nb_chunks_per_tank,
        const int64_t chunk_size, const torch::Device device)
        : nb_tanks(nb_tanks), chunk_size(chunk_size), device(device),
          step_offsets(torch::arange(chunk_size)),
          valid_counts(valids.reshape({nb_chunks_per_tank, chunk_size, nb_tanks}).sum(1).flatten()),
          valid_units(torch::nonzero(valid_counts > 0).squeeze(-1)) {}

    int64_t LiquidPpoTrainer::ChunkIndex::units_per_batch(const int64_t batch_size) const {
        return std::max<int64_t>(1, batch_size / chunk_size);
    }

    std::tuple<torch::Tensor, torch::Tensor>
    LiquidPpoTrainer::ChunkIndex::rows(const torch::Tensor &units) const {
        const auto tank = units.remainder(nb_tanks);
        const auto start_step = units.div(nb_tanks, "floor") * chunk_size;
        return {
            ((start_step.unsqueeze(1) + step_offsets.unsqueeze(0)) * nb_tanks + tank.unsqueeze(1))
                .flatten()
                .to(device),
            (start_step * nb_tanks + tank).to(device)};
    }

    float LiquidPpoTrainer::ChunkIndex::live_rows(const torch::Tensor &units) const {
        return valid_counts.index_select(0, units).sum().item<float>();
    }

    float
    LiquidPpoTrainer::ChunkIndex::batch_live_rows(const std::vector<torch::Tensor> &blocks) const {
        return std::accumulate(
            blocks.begin(), blocks.end(), 0.f,
            [this](const float sum, const torch::Tensor &units) { return sum + live_rows(units); });
    }

    void LiquidPpoTrainer::train() {
        const auto device = actor->parameters().back().device();

        auto rollout = rollout_buffer->get_rollout();

        rollout.states = rollout.states.to(device);

        set_train(false);
        const auto gae = compute_gae(rollout, device);

        const auto nb_steps = rollout.rewards.size(0);
        const auto nb_tanks = rollout.rewards.size(1);

        const auto nb_chunks_per_tank = nb_steps / chunk_size;
        if (nb_chunks_per_tank == 0) return;
        const auto nb_kept_steps = nb_chunks_per_tank * chunk_size;

        const auto rows = flat_rows(rollout, gae, nb_kept_steps, device);

        const ChunkIndex chunks(
            rollout.valids.slice(0, 0, nb_kept_steps), nb_tanks, nb_chunks_per_tank, chunk_size,
            device);
        if (chunks.valid_units.size(0) == 0) return;

        set_train(true);
        run_actor_epochs(rows, chunks);
        run_critic_epochs(rows, chunks);
        set_train(false);
    }

    LiquidPpoTrainer::FlatRows LiquidPpoTrainer::flat_rows(
        const LiquidPpoRollout &rollout, const LiquidGaeResult &gae, const int64_t nb_kept_steps,
        const torch::Device device) {
        // flat [T_kept * nb_tanks, ...] rows, all on the device
        const auto flat = [&](const torch::Tensor &tensor) {
            return flatten_steps(tensor.slice(0, 0, nb_kept_steps)).to(device);
        };
        return {
            .vision = flat(rollout.states.get_vision()),
            .proprioception = flat(rollout.states.get_state()),
            .continuous_pre_tanh = flat(rollout.continuous_pre_tanhs),
            .discrete_actions = flat(rollout.actions.discrete_action),
            .fire_masks = flat(rollout.states.get_fire_mask()),
            .old_continuous_log_probs = flat(rollout.continuous_log_probs),
            .old_discrete_log_probs = flat(rollout.discrete_log_probs),
            .advantages = flat(gae.advantages),
            .aim_advantages = flat(gae.aim_advantages),
            .returns = flat(gae.returns),
            .valids = flat(rollout.valids),
            .actor_hiddens = flat(rollout.actor_hiddens),
            .critic_hiddens = flat(gae.critic_hiddens)};
    }

    LiquidSigmaGradients LiquidPpoTrainer::sigma_gradients() {
        const auto device = actor->parameters().back().device();

        auto rollout = rollout_buffer->get_rollout();
        rollout.states = rollout.states.to(device);

        set_train(false);
        const auto gae = compute_gae(rollout, device);

        const auto nb_steps = rollout.rewards.size(0);
        const auto nb_tanks = rollout.rewards.size(1);
        const auto nb_chunks_per_tank = nb_steps / chunk_size;
        const auto nb_kept_steps = nb_chunks_per_tank * chunk_size;

        if (nb_chunks_per_tank == 0) return {};

        const auto rows = flat_rows(rollout, gae, nb_kept_steps, device);
        const ChunkIndex chunks(
            rollout.valids.slice(0, 0, nb_kept_steps), nb_tanks, nb_chunks_per_tank, chunk_size,
            device);

        std::vector<torch::Tensor> row_ids, surrogates, entropies, mus, sigmas, pre_tanhs,
            advantages, aim_advantages, fire_masks, fires;

        set_train(true);
        const auto units_per_block = chunks.units_per_batch(actor_gpu_batch_size);
        for (const auto &units: chunks.valid_units.split(units_per_block)) {
            const auto [unit_rows, start_rows] = chunks.rows(units);
            const auto block = rows.select_chunks(unit_rows, start_rows, chunk_size);
            const auto terms = actor_terms(block);

            // per row: d(row term)/d(sigma) * sigma = d(row term)/d(log sigma)
            const auto d_surrogate = torch::autograd::grad(
                {terms.surrogate.sum()}, {terms.sigma}, {}, /*retain_graph=*/true)[0];
            const auto d_entropy =
                torch::autograd::grad({terms.continuous_bonus.sum()}, {terms.sigma})[0];

            const auto valid_rows = [&](const torch::Tensor &tensor) {
                return tensor.flatten(0, 1).index_select(0, terms.valid_idx).detach().cpu();
            };

            row_ids.push_back(unit_rows.index_select(0, terms.valid_idx).cpu());
            surrogates.push_back((d_surrogate * terms.sigma).detach().cpu());
            entropies.push_back((d_entropy * terms.sigma).detach().cpu());
            mus.push_back(terms.mu.detach().cpu());
            sigmas.push_back(terms.sigma.detach().cpu());
            pre_tanhs.push_back(valid_rows(block.continuous_pre_tanh));
            advantages.push_back(valid_rows(block.advantages).squeeze(-1));
            aim_advantages.push_back(valid_rows(block.aim_advantages).squeeze(-1));
            fire_masks.push_back(valid_rows(block.fire_masks).select(-1, 0));
            fires.push_back(valid_rows(block.discrete_actions).select(-1, 0));
        }
        set_train(false);

        const auto ids = torch::cat(row_ids);
        return {
            .steps = ids.div(nb_tanks, "floor"),
            .tanks = ids.remainder(nb_tanks),
            .surrogate = torch::cat(surrogates),
            .entropy = torch::cat(entropies),
            .mu = torch::cat(mus),
            .sigma = torch::cat(sigmas),
            .pre_tanh = torch::cat(pre_tanhs),
            .advantages = torch::cat(advantages),
            .aim_advantages = torch::cat(aim_advantages),
            .fire_masks = torch::cat(fire_masks),
            .fires = torch::cat(fires)};
    }

    void LiquidPpoTrainer::run_actor_epochs(const FlatRows &rows, const ChunkIndex &chunks) const {
        int nb_actor_steps = 0;
        int nb_planned_actor_steps = 0;
        bool kl_exceeded = false;

        for (int e = 0; e < actor_epochs; e++) {
            const auto batches = shuffled_batches(
                chunks.valid_units, chunks.units_per_batch(actor_batch_size),
                chunks.units_per_batch(actor_gpu_batch_size));
            nb_planned_actor_steps += static_cast<int>(batches.size());

            for (const auto &blocks: batches) {
                if (kl_exceeded) break;

                actor_optim->zero_grad();
                alphas->zero_grad();

                const auto batch_rows = chunks.batch_live_rows(blocks);

                std::vector<torch::Tensor> kl_per_rows;
                for (const auto &units: blocks) {
                    const auto [unit_rows, start_rows] = chunks.rows(units);
                    kl_per_rows.push_back(backward_actor(
                        rows.select_chunks(unit_rows, start_rows, chunk_size),
                        chunks.live_rows(units) / batch_rows));
                }

                // KL of the policy this step starts from
                const auto approx_kl = trimmed_kl(kl_per_rows);
                kl_metric->add(approx_kl);

                kl_exceeded = target_kl > 0.f && approx_kl > 1.5f * target_kl;
                if (kl_exceeded) break;

                torch::nn::utils::clip_grad_norm_(actor->parameters(), grad_norm_max);
                actor_optim->step();
                alphas->step();
                nb_actor_steps++;
            }
        }

        skip_fraction_metric->add(
            1.f - static_cast<float>(nb_actor_steps) / static_cast<float>(nb_planned_actor_steps));
    }

    void LiquidPpoTrainer::run_critic_epochs(const FlatRows &rows, const ChunkIndex &chunks) const {
        for (int e = 0; e < critic_epochs; e++)
            for (const auto &blocks: shuffled_batches(
                     chunks.valid_units, chunks.units_per_batch(critic_batch_size),
                     chunks.units_per_batch(critic_gpu_batch_size))) {
                critic_optim->zero_grad();

                const auto batch_rows = chunks.batch_live_rows(blocks);

                for (const auto &units: blocks) {
                    const auto [unit_rows, start_rows] = chunks.rows(units);
                    backward_critic(
                        rows.select_chunks(unit_rows, start_rows, chunk_size),
                        chunks.live_rows(units) / batch_rows);
                }

                torch::nn::utils::clip_grad_norm_(critic->parameters(), grad_norm_max);
                critic_optim->step();
            }
    }

    LiquidPpoTrainer::ActorTerms LiquidPpoTrainer::actor_terms(const FlatRows &block) const {
        const auto out =
            actor->act_sequence(block.vision, block.proprioception, block.actor_hiddens);

        // recurrence done: fold time into rows and keep the live transitions only
        const auto valid_idx = torch::nonzero(block.valids.flatten(0, 1).squeeze(-1)).squeeze(-1);
        const auto rows = [&](const torch::Tensor &tensor) {
            return tensor.flatten(0, 1).index_select(0, valid_idx);
        };

        const auto mu = rows(out.mu);
        const auto sigma = rows(out.sigma);
        const auto discrete_proba = rows(out.discrete);
        const auto discrete_mask =
            TorchState::discrete_action_mask(rows(block.fire_masks), discrete_proba.size(-1));

        const auto current = PolicyLogProbs::split(
            gaussian_tanh_log_pdf(rows(block.continuous_pre_tanh), mu, sigma),
            (bernoulli_log_proba(rows(block.discrete_actions), discrete_proba) * discrete_mask)
                .sum(-1, true));
        const auto old = PolicyLogProbs::split(
            rows(block.old_continuous_log_probs), rows(block.old_discrete_log_probs));

        const auto [surrogate, log_ratio, ratio] = clipped_surrogate(
            current, old, rows(block.aim_advantages), rows(block.advantages), clip_epsilon);

        const auto continuous_entropy = gaussian_tanh_entropy(mu, sigma);
        const auto discrete_entropy = bernoulli_entropy(discrete_proba) * discrete_mask;

        // the alphas are constants for the actor: they learn from their own dual loss
        return {
            .valid_idx = valid_idx,
            .mu = mu,
            .sigma = sigma,
            .discrete_mask = discrete_mask,
            .ratio = ratio,
            .log_ratio = log_ratio,
            .continuous_entropy = continuous_entropy,
            .discrete_entropy = discrete_entropy,
            .surrogate = surrogate,
            .continuous_bonus = alphas->continuous().detach() * continuous_entropy,
            .discrete_bonus = alphas->discrete().detach() * discrete_entropy};
    }

    torch::Tensor
    LiquidPpoTrainer::backward_actor(const FlatRows &block, const float batch_fraction) const {
        const auto terms = actor_terms(block);

        const auto entropy_bonus =
            terms.continuous_bonus.sum(-1, true) + terms.discrete_bonus.sum(-1, true);

        const auto actor_loss = -torch::mean(terms.surrogate + entropy_bonus);

        // weighted so that the accumulated gradient is the one of the whole rollout's mean
        (actor_loss * batch_fraction).backward();
        alphas->backward(
            terms.continuous_entropy, terms.discrete_entropy, terms.discrete_mask, batch_fraction);

        // metrics
        actor_mean_loss_metric->add(actor_loss.cpu().item<float>());
        continuous_entropy_metric->add(terms.continuous_entropy.mean().item<float>());
        // mean over the discrete actions that had an effect
        discrete_entropy_metric->add(
            (terms.discrete_entropy.sum() / terms.discrete_mask.sum().clamp_min(1.f))
                .item<float>());

        clip_fraction_metric->add(
            ((terms.ratio - 1.f).abs() > clip_epsilon).to(torch::kFloat).mean().item<float>());

        return (terms.ratio - 1.f - terms.log_ratio).detach().flatten();
    }

    void
    LiquidPpoTrainer::backward_critic(const FlatRows &block, const float batch_fraction) const {
        const auto [value, next_x] =
            critic->value_sequence(block.vision, block.proprioception, block.critic_hiddens);

        // recurrence done: fold time into rows and keep the live transitions only
        const auto valid_idx = torch::nonzero(block.valids.flatten(0, 1).squeeze(-1)).squeeze(-1);
        const auto values = value.flatten(0, 1).index_select(0, valid_idx);
        const auto valid_returns = block.returns.flatten(0, 1).index_select(0, valid_idx);

        const auto critic_loss = torch::mse_loss(values, valid_returns, at::Reduction::Mean);

        (critic_loss * batch_fraction).backward();

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

            const auto [value, next_x] = critic->value(
                rollout.states.get_vision()[t].to(device), rollout.states.get_state()[t].to(device),
                critic_carry);

            value_list.push_back(value.cpu());
            critic_carry = next_x;
        }

        const auto values = torch::stack(value_list, 0);
        const auto critic_hiddens = torch::stack(hidden_list, 0);

        const auto bootstrap_value =
            critic
                ->value(
                    rollout.bootstrap_state.get_vision().to(device),
                    rollout.bootstrap_state.get_state().to(device), critic_carry)
                .value.cpu()
                .unsqueeze(0);

        // next values are the values shifted by one step, closed by the bootstrap state
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
            .returns = advantages + values,
            .critic_hiddens = critic_hiddens};
    }

    std::vector<std::shared_ptr<AbstractMetric>> LiquidPpoTrainer::get_metrics() {
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

    std::vector<std::shared_ptr<AbstractMetric>> LiquidPpoTrainer::get_csv_only_metrics() {
        return alphas->get_metrics();
    }

    void LiquidPpoTrainer::save(const std::filesystem::path &output_folder) {
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

    void LiquidPpoTrainer::load(const std::filesystem::path &input_folder) const {
        const auto device = actor->parameters().back().device();

        load_torch(input_folder, actor, "actor.pt");
        load_torch(input_folder, critic, "critic.pt");
        load_torch(input_folder, return_scaler, "return_scaler.pt");
        alphas->load(input_folder);

        to(device);
    }

    void LiquidPpoTrainer::set_train(const bool train) const {
        actor->train(train);
        critic->train(train);
    }

    void LiquidPpoTrainer::to(const torch::Device device) const {
        actor->to(device);
        critic->to(device);
        alphas->to(device);
    }

    int LiquidPpoTrainer::count_parameters() {
        return count_parameters_impl(actor->parameters())
               + count_parameters_impl(critic->parameters()) + alphas->count_parameters();
    }
}// namespace arenai::agent
