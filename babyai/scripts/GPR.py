import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

SR = np.array([0.73, 0.73, 0.73, 0.73,
               0.66, 0.66, 0.66, 0.66,
               0.56, 0.56, 0.56, 0.56,
               0.41, 0.41, 0.41, 0.41,
               0.250, 0.250, 0.250, 0.250]).reshape(-1, 1)
SE = np.array([0.560, 0.548, 0.573, 0.556,
               0.555, 0.556, 0.568, 0.577,
               0.557, 0.563, 0.529, 0.538,
               0.501, 0.488, 0.452, 0.481,
               0.192, 0.214, 0.206, 0.132]).reshape(-1, 1)
print(SR)
print(SE)
kernel = RBF(length_scale_bounds=(1e-05, 100000.0))

"""alpha_step = np.arange(1e-4, 5e-3, 2e-4)
score = np.zeros_like(alpha_step)
for a in range(len(alpha_step)):
    gpr = GaussianProcessRegressor(alpha=alpha_step[a], kernel=kernel, random_state=0).fit(SR, SE)
    score[a] = gpr.score(SR, SE)

plt.plot(alpha_step, score)
plt.show()"""

gpr = GaussianProcessRegressor(alpha=0.0005, kernel=kernel, random_state=0).fit(SR, SE)
print(gpr.score(SR, SE))
SR_pred = np.arange(0.250, 0.73, 0.001)
SE_mean, SE_std = gpr.predict(SR_pred.reshape(-1, 1), return_std=True)
SE_mean = SE_mean.reshape(480, )
plt.scatter(SR, SE, label="Observations")
plt.plot(SR_pred, SE_mean)
plt.fill_between(SR_pred,
                 SE_mean + 1.96 * SE_std,
                 SE_mean - 1.96 * SE_std,
                 alpha=0.5,
                 label=r"95% confidence interval")
plt.ylabel("Sample Efficiency")
plt.xlabel("Success rate of the QA")
plt.legend()
plt.show()

high_curve = SE_mean + 1.96 * SE_std
valid_idx = np.where(high_curve >= 0.5)[0][0]
print(SR_pred[valid_idx])

low_curve = SE_mean - 1.96 * SE_std
valid_idx = np.where(low_curve >= 0.5)[0][0]
print(SR_pred[valid_idx])
"""print(gpr.get_params(deep=True))
SR_min = np.arange(0.250, 0.73, 0.001)
proba_SR_min = []
len_SR_min = len(SR_min)

print("proba inferior: {}".format(len(SE_pred[SE_pred < 0.5])/len(SE_pred)))

for i in range(1, 481):
    proba_inferior = (len(SE_pred[SE_pred < 0.5])/len(SE_pred))**(i-1)
    proba_superior = len(SE_pred[SE_pred > 0.5])/len(SE_pred)
    proba_SR_min.append(proba_inferior*proba_superior)
len(SR_min)
print(len(proba_SR_min))
plt.plot(SR_min, np.array(proba_SR_min))
plt.show()"""
