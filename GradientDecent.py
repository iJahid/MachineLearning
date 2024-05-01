import numpy as np


def gradient_decent(x, y):
    m_curr = b_curr = 0
    n = len(x)

    learing_rate = 0.08
    # why 0.08 its for tuning (trial and error . see below cost output comments)
    iterations = 100
    # why 100 its for tuning (trial and error . see below cost output comments)

    for i in range(iterations):
        y_predicted = m_curr*x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        # m is partial derivative from theory (.png file)
        md = -(2/n)*sum(x*(y-y_predicted))
        # b is partial derivative from theory (.png file)
        bd = -(2/n)*sum(y-y_predicted)
        # adjust m_curr using therory(.png file)
        m_curr = m_curr-(learing_rate*md)
        # adjust b_curr using therory(.png file)
        b_curr = b_curr-(learing_rate*bd)

        # cost is reducing in the output
        # target : adjust learning_rate and iteration to get when the cost is up
        # .. so the last learning rate of cost readcing is the best
        # here  learning rate=0.001, 0.01 to 0.08 is reducing but .09 is increasing
        # so the best learning rate is 0.08
        print("m {} b {} cost {} iteration {}".format(m_curr, b_curr, cost, i))


x = np.array([1, 2, 3, 4, 5])
y = np.array([5, 7, 9, 11, 13])

print(gradient_decent(x, y))
