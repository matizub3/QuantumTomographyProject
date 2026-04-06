import matplotlib.pyplot as plt
import pickle

import numpy as np

q_flow_params, q_losses_mean, q_losses_std = pickle.load(open("sample_complexity_results/CNF_cat_FFJORD_Adaptive_layer_size=10_KL_uniform_num_samples=500_epochs=1000_cosine_decay_warmup=0.1_5_15_3.dat", "rb"))
qst_samples, qst_losses_mean, qst_losses_std = pickle.load(open("baselines/sample_efficiency/qst_cat.pkl", "rb"))

#q_flow_params, q_losses_mean, q_losses_std = pickle.load(open("sample_complexity_results/CNF_num_0_FFJORD_Adaptive_layer_size=10_KL_uniform_num_samples=200_epochs=1000_cosine_decay_warmup=0.1_5_15_3.dat", "rb"))
#qst_samples, qst_losses_mean, qst_losses_std = pickle.load(open("baselines/sample_efficiency/qst_num.pkl", "rb"))


qst_samples = np.array(qst_samples) ** 2
print(qst_samples)
print(qst_losses_mean)
print(qst_losses_std)

q_flow_samples = np.array(q_flow_params) ** 2

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))

for key in qst_losses_mean[0]:
    axes[0].plot(qst_samples, [item[key] for item in qst_losses_mean], label = f"QST {key}")
    axes[0].fill_between(
        qst_samples, 
        np.array([item[key] for item in qst_losses_mean]) - np.array([item[key] for item in qst_losses_std]),
        np.array([item[key] for item in qst_losses_mean]) + np.array([item[key] for item in qst_losses_std]),
        alpha = 0.3,
    )

for key in q_losses_mean:
    axes[1].plot(q_flow_samples, q_losses_mean[key], label = f"Q {key}")
    axes[1].fill_between(
        q_flow_samples, 
        np.array(q_losses_mean[key]) - np.array(q_losses_std[key]),
        np.array(q_losses_mean[key]) + np.array(q_losses_std[key]),
        alpha = 0.3,
    )

axes[0].legend()
axes[1].legend()
plt.show()