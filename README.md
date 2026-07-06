## 1. Top-level Python files

### `main.py`
This is the main command-line entrypoint for the whole project. It reads the command-line flags, decides which representation and target problem to run, builds the requested flow model, and launches training.

Conceptually, `main.py` is the "experiment controller." It does not contain all the mathematics itself; instead, it wires together the argument parser, the target distribution, the loss, the optimizer setup, and the training loop. When you run a command such as `python main.py --problem cat ...`, this is the file that starts everything.

### `parse_args.py`
This file defines the command-line interface for the project. It maps short and long flags such as problem choice, activation function, number of layers, loss type, plotting frequency, and representation (`Q` versus `W`) into a structured argument object used by the rest of the code.

It is important because the paper's experiments are largely encoded as different argument settings. In practice, this file determines how launch scripts such as `Q_1_well.sh` or `W_experimental.sh` translate paper hyperparameters into actual code behavior.

### `training.py`
This is the main training loop for the standard experiments. It takes the assembled model, the selected target distribution, and the selected loss, then performs iterative optimization.

This file is where the project spends most of its runtime. Each epoch/iteration it draws model samples, computes the quantum loss, differentiates that loss through the flow, updates parameters with the chosen optimizer, and records information for visualization and checkpointing.

### `training_sampler.py`
This is a variant of the training code used when the experiment relies on a sample bank or sample-efficient estimation strategy. Instead of always recomputing everything from scratch, it can work with sampled points that are reused or managed in a separate way.

This file matters because some of the paper's objectives are explicitly designed to be more sample-efficient. If a launch script points into the sampler-based path, this is the file that implements that alternative training workflow.

### `flow_IO.py`
This file handles saving and loading flow checkpoints and related serialized artifacts. After a long run, the learned flow parameters need to be stored so they can be reused for evaluation, figure generation, or continuation of training.

In other words, `flow_IO.py` is the persistence layer for trained models. Without it, you would have to retrain from scratch every time you wanted to regenerate figures or inspect a learned state.

### `plotting.py`
This file creates the visualizations used during and after training. It typically handles losses over time, learned densities, comparisons between target and reconstructed functions, and other summary plots.

It is important because the paper compares exact and reconstructed quantum phase-space functions visually. Even when the training itself is correct, you still need `plotting.py` to interpret what the learned flow is doing.

### `video.py`
This file is for producing animation or time-evolution style visual outputs from training checkpoints or intermediate states. It is not essential to the mathematical core of training, but it is useful for making the training dynamics interpretable.

Think of it as a convenience layer for presentation rather than a dependency for the optimizer itself.

### `sample_efficiency_test.py`
This script is used to probe how performance changes as the amount of sampled information varies. It is more of an experiment driver than a core library module.

Its role is to support claims about how efficiently the flow-based approach learns under different sampling budgets. If you are reproducing auxiliary analyses rather than just the main figures, this becomes relevant.

### `sample_complexity_test.py`
This script studies how reconstruction behavior changes as the problem size or data budget scales. Like the previous file, it is an analysis/experiment wrapper rather than a core dependency of every training run.

Use this when you want to quantify scaling behavior rather than just train a single model.

---

## 2. Flow model code (`Q_flows/`)

### `Q_flows/continuous.py`
This file contains the implementation of the continuous normalizing flow itself. It defines how a base distribution is pushed through an ODE-defined transformation to produce the learned distribution in phase space.

This is one of the mathematically central files in the repo. The flow model needs to produce transformed samples and track the corresponding change in log density, and this file is where that continuous-time transport machinery lives.

### `Q_flows/flow_util.py`
This file provides helper functions used by the flow implementation. That usually includes utilities for applying the flow, managing shapes, handling density bookkeeping, or interfacing the flow code with the rest of the training stack.

You can think of it as the support code that makes the main continuous flow implementation usable inside experiments.

---

## 3. Distribution and target-state code (`distributions/`)

### `distributions/distribution.py`
This file provides shared abstractions or base behavior for target distributions. It defines the interface that concrete Q-function and Wigner-function targets are expected to satisfy.

Its purpose is organizational as much as mathematical: it lets the training loop treat many different target states in a consistent way.

### `distributions/density_targets.py`
This file collects target-density definitions or utilities shared by several problems. It is essentially a catalog of the benchmark targets the repo knows how to generate.

It is useful because the paper compares reconstruction quality across multiple states, and those states need to be defined somewhere in a common format.

### `distributions/Q_function.py`
This file contains code specific to the Husimi Q-function representation. It defines how the target density or objective should be evaluated when the experiment is formulated in Q-space.

If a run is using the default Q-based formulation from the paper's synthetic experiments, this file is on the critical path.

### `distributions/W_function.py`
This file plays the same role for the Wigner-function representation. It defines the evaluation machinery needed when the learned object is the Wigner function instead of the Q function.

This file is especially relevant for the synthetic and experimental Wigner reconstructions.

### `distributions/Q_targets.py`
This file defines concrete benchmark target states in the Q representation. Based on the project layout and launch scripts, these include states such as cat states, Fock/number states, binomial states, GKP states, and multi-well tensor-product constructions.

The training loop relies on this file to know what "ground truth" it should approximate in a given Q-based experiment.

### `distributions/W_targets.py`
This file defines the corresponding benchmark states for the Wigner representation. These targets are needed whenever the paper or scripts request a W-space reconstruction.

The separation between `Q_targets.py` and `W_targets.py` keeps representation-specific details cleanly separated, which is helpful because the same physical state may look quite different under Q and W descriptions.

---

## 4. Loss code (`losses/`)

### `losses/base_loss.py`
This file provides shared loss infrastructure. It defines the interface and common behavior used by specific Q and W loss implementations.

The rest of the loss files inherit from or build on this logic so that the training loop can use different objectives interchangeably.

### `losses/multiple_loss.py`
This file combines or manages multiple loss terms. It is useful when an experiment needs a compound objective or when several components must be aggregated into one scalar value for optimization.

In practice, this file makes the codebase flexible: the training loop can ask for "the loss" without hard-coding every individual term itself.

### `losses/Q_KL.py`
This file implements a KL-style loss for Q-function learning. In the paper's Q-flow experiments, the main objective is a forward KL-like/sample-efficient Q-based loss, so this file is one of the central places where that objective is actually computed.

From an optimization perspective, this loss pushes the learned flow distribution toward the target Q distribution by penalizing mismatch in log-density space.

### `losses/Q_KL_rev.py`
This file implements the reverse-KL variant for the Q-function setting. Reverse KL tends to behave differently from forward KL, often favoring sharper mode-seeking behavior.

Even if the main paper emphasizes one preferred loss, this file gives the project an alternative Q-space objective for ablations or exploratory runs.

### `losses/Q_L1.py`
This file implements an L1 discrepancy for Q-space quantities. Instead of comparing through KL divergence, it compares through absolute error.

This objective is useful when you care about direct pointwise mismatch more than probabilistic divergence language.

### `losses/W_loss.py`
This file implements the loss used for Wigner-function training. In the paper and surrounding project notes, W-space experiments are associated with an **L1-style reconstruction objective**, so this file is the main place where Wigner mismatch is measured and turned into something the optimizer can minimize.

Because the Wigner function can be oscillatory and sign-changing, a direct reconstruction-oriented loss is a natural choice here.

---

## 5. Training setup code (`training_setup/`)

### `training_setup/setup.py`
This file is the high-level setup dispatcher. It takes parsed arguments and assembles the full experiment configuration: target distribution, flow architecture, optimizer/loss pair, plotting schedule, and other settings.

If `main.py` is the conductor, `setup.py` is the stage manager. It builds the objects that the training loop needs before optimization begins.

### `training_setup/distribution_setup.py`
This file maps the command-line problem name (for example `cat`, `num_0`, `GKP`, `all3`, or an experimental Wigner dataset) into an actual target distribution object.

This is where the abstract problem label gets turned into real physics/math. If you ever want to add a new benchmark state, this is one of the first files you would extend.

### `training_setup/flow_setup.py`
This file constructs the actual flow model according to the chosen hyperparameters: hidden layer size, number of layers, activation function, network variant, and possibly solver-related details.

It is the bridge between user-selected architecture flags and an instantiated learnable CNF.

### `training_setup/loss_setup.py`
This file selects and instantiates the requested loss object. Based on the argument string, it decides whether the run should use a Q-KL loss, reverse KL, L1, a model-control objective, or a Wigner-specific loss.

This separation keeps `training.py` cleaner, because the training loop only has to call the loss object it is given.

---


## 6. Primary run scripts (`scripts/final/`)

### `scripts/final/Q_1_well.sh`
This shell script launches the one-well Q-flow experiments for the standard benchmark states. It is essentially a batch wrapper around several `python main.py ...` commands.

This is one of the easiest ways to reproduce the paper's Q-space single-well results, especially on a cluster.

### `scripts/final/W_synthetic.sh`
This script launches the synthetic Wigner experiments. It bundles together the W-space runs for states such as cat, number/Fock, binomial, and GKP.

If your target is the paper's Wigner synthetic reconstructions, this is a convenient starting point.

### `scripts/final/W_experimental.sh`
This script launches the experimental Wigner reconstruction run used for the comparison against the QST-CGAN baseline.

Among the scripts in the repo, this is one of the most directly tied to a specific paper figure.

### `scripts/final/Q_all.sh`
This script launches the multi-well Q experiment corresponding to the five-well scalability setting.

### `scripts/final/Q_multi_num.sh`
This script launches a larger multi-well or multi-number-state experiment, such as the ten-well case discussed in the extended results.

---

## 7. Losses and evaluation metrics used in the code

The repository supports more than one loss, but the paper-facing code path is easiest to understand if you separate **training objective** from **evaluation metric**.

### A. Training objectives
The codebase includes at least the following objective families:

- **Q-space KL objective** via `losses/Q_KL.py`  
  Used for Q-flow training. This is the main probabilistic divergence-style objective in the Husimi Q setting.

- **Q-space reverse KL objective** via `losses/Q_KL_rev.py`  
  An alternative Q-flow objective used for ablations or exploratory comparisons.

- **Q-space L1 objective** via `losses/Q_L1.py`  
  A direct absolute-error objective in the Q representation.

- **W-space reconstruction loss** via `losses/W_loss.py`  
  Used for Wigner-function training. Project notes and script naming indicate that this is an L1-style reconstruction loss in W-space.

- **Composite/multi-term losses** via `losses/multiple_loss.py`  
  Used when the code needs to bundle several terms together.

### B. Evaluation metrics
The most natural evaluation quantities in this repo are reconstruction errors between the learned and target phase-space functions. In practice, that means:

- **Pointwise L1 error** between target and reconstructed function values  
- **Loss value over time** as an optimization diagnostic  
- **Visual agreement** between exact and reconstructed Q or W functions  
- In some experiment wrappers, **sample-efficiency / scaling curves** as the computational evaluation output

A good mental model is:
- **KL-like losses** are used to *train* Q-flows.
- **L1-style discrepancy** is used both as a Wigner training objective and as a natural *evaluation metric* for reconstruction quality.
- Visual plots are a major part of evaluation because the learned object is a function over phase space, not just a class label.

---

## 8. Thorough explanation of the main training loop

This section explains what `training.py` / `training_sampler.py` are doing conceptually.

### Step 1: Parse the experiment configuration
The run begins in `main.py`, which reads command-line options such as:
- which target state to reconstruct,
- whether to work in Q space or W space,
- what neural ODE / continuous-flow architecture to use,
- how wide and deep the network should be,
- what loss to optimize,
- how often to plot and save.

These settings are then passed into the `training_setup` package.

### Step 2: Build the target distribution
`training_setup/distribution_setup.py` converts the symbolic problem name into a concrete target object. For example, `cat`, `num_0`, `binom_0`, `GKP`, or `all3` gets mapped to code that knows how to evaluate the target phase-space function.

At this point the code has a mathematically defined "ground truth" object that it wants the flow to learn.

### Step 3: Build the flow model
`training_setup/flow_setup.py` constructs the continuous normalizing flow. The model starts from a simple base distribution and learns a continuous transformation that pushes that base into something matching the target quantum distribution.

This flow is parameterized by a neural network embedded in an ODE. During training, the model not only generates transformed samples but also tracks the log-density correction required by the change of variables.

### Step 4: Select the loss
`training_setup/loss_setup.py` chooses the objective based on the experiment. Q experiments typically use a KL-based objective; W experiments use a reconstruction-oriented Wigner loss.

From the training loop's perspective, this selected loss object is just a function: give it the current model state and some samples, and it returns a scalar loss plus any useful diagnostics.

### Step 5: Initialize optimizer state
Before the optimization loop begins, the code initializes the trainable flow parameters and the optimizer state. Depending on the implementation details, that can include learning-rate schedules, warmup behavior, or internal states needed by an optimizer such as Adam/Optax.

This is the point where the model is ready to learn but has not yet seen any target-driven gradient information.

### Step 6: Draw samples from the current flow
At each iteration, the training code samples latent points from the base distribution and pushes them through the continuous flow. This produces samples in the target phase space together with density information.

If the experiment uses the sampler-based path, `training_sampler.py` may manage a bank of points or a more sample-efficient reuse strategy rather than generating everything in the simplest possible way each time.

### Step 7: Evaluate the target and compute the loss
Now the code compares the current flow-induced distribution to the target quantum object.

- In a **Q-space KL** run, the code evaluates how different the learned Q distribution is from the target Q distribution using the KL-style objective.
- In a **W-space** run, it measures reconstruction mismatch using the Wigner loss, which is effectively L1-style in the project notes and script descriptions.

This step produces a single scalar objective that says how wrong the current flow is.

### Step 8: Differentiate through the flow
The code backpropagates the scalar loss through the continuous normalizing flow. Because the flow is defined through an ODE, this means differentiating through the solver-defined transformation and the density-tracking terms.

This is the technically heavy part of the project: the code has to propagate gradient information not only through a neural network, but through the entire continuous transport mechanism.

### Step 9: Update parameters
The optimizer takes the gradient and updates the flow parameters. After this update, the base distribution will be transported slightly differently on the next iteration.

Over many iterations, these small updates gradually sculpt the learned distribution so that it resembles the target state more and more closely.

### Step 10: Record diagnostics
The loop records things such as:
- current loss,
- possibly reconstruction error,
- sample-based summaries,
- timing or efficiency statistics.

These diagnostics make it possible to see whether training is stable, whether the model is converging, and whether hyperparameters need adjustment.

### Step 11: Plot and checkpoint periodically
Every few iterations or epochs, the code uses `plotting.py` to visualize progress and `flow_IO.py` to save the current model. This is crucial for long-running experiments because the paper figures are produced from these saved artifacts.

Without checkpointing, a long run interrupted near the end would have to be restarted from scratch.

### Step 12: Final evaluation and figure generation
Once training is complete, the saved checkpoints and recorded metrics are used to generate the final plots and paper figures. The `figure_plotting` scripts typically enter here.

So the full training story is:

1. choose a target state,  
2. build a CNF,  
3. choose a Q- or W-based loss,  
4. repeatedly sample → compare → differentiate → update,  
5. save plots/checkpoints,  
6. generate final figures.

That is the central algorithmic loop behind the repository.

---


## 9. How to run the code

Open a terminal in the repository root and activate your environment. Then start with one of the paper-style commands.

Example single run:
```bash
python main.py --problem cat -b 1 -pe 10 -a SINE -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_efficient --rescale True -w 0.1 -d True
```

Example experimental Wigner run:
```bash
python main.py --problem QST_CGAN_W_Neg -r W -b 1 -pe 10 -a GELU -res True -v False -net FFJORDNet -ls 30 -nl 5 -l KL_model_control --rescale True -e 1000 -d True -n 1000 -w 0.1
```

If you are on a cluster, the shell scripts in `scripts/final/` are the most convenient way to launch the main experiment families.

---

## 10. Final summary

The repository is built around a clean separation of concerns:
- `main.py` handles experiment entry,
- `training_setup/*` assembles the experiment,
- `Q_flows/*` defines the CNF model,
- `distributions/*` defines the target quantum states,
- `losses/*` defines what is optimized,
- `training.py` / `training_sampler.py` perform optimization,
- `plotting.py` and `figure_plotting/*` turn results into interpretable outputs.
