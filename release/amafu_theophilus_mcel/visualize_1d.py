# some simple code snippet to illustrate how to plot a function
# you should modify y_ev

import matplotlib.pyplot as plt
import numpy as np

# X_train, X_test, t_train, t_test should all be 1-d, and need to be defined as well 

# plot a curve showing learned function
x_ev = np.arange(min(X_n), max(X_n) + 0.1, 0.1)
y_ev = ..... # put your regression estimate here

plt.plot(x_ev, y_ev, 'r.-')
plt.plot(X_train, t_train, 'gx', markersize=10)
plt.plot(X_test, t_test, 'bo', markersize=10, mfc='none')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Fig degree %d polynomial' % 5)
