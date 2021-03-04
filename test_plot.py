import matplotlib.pyplot as plt
from numpy import linspace

res = [0.1,0.3,0.7,0.8,0.85,0.88,0.89, 0.99, 0.999]
x = linspace(0, 1, num=len(res), endpoint=True)
x2 = linspace(0, 100, num=len(res), endpoint=True)
plt.plot(x, res, color='blue', label='Train Loss Over Epoches')
plt.savefig('test.png', dpi=300)
plt.clf()
plt.plot(x2, res, scalex=1000)
plt.figtext(0.5,0.01, "OK", ha='center')
plt.savefig('test2.png', dpi=300)