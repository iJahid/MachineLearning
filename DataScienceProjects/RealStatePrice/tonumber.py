import pandas as p
import numpy as np


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


def getDigit(x):
    if (is_float(x)):
        return float(x)
    else:
        # sqmeter_to_foot = 10.7639
        # sqyard_to_foot = 9
        # parch_to_foot = 272.25
        # multyplier = 1
        # if "Meter" in x:
        #     multyplier = sqmeter_to_foot
        # elif "Yards" in x:
        #     multyplier = sqyard_to_foot
        # elif "Perch" in x:
        #     multyplier = parch_to_foot
        # elif "-" in x:
        #      return (int(x.split('-')[0])+int(x.split('-')[1]))/2
        # else:
        d = [*x]
        n = ''
        for i in d:
            if (is_float(i)):
                n = n+str(int(i))
            else:
                return int(n)
    return n


print(getDigit('3gdsf56'))
x = "2312 - 123123"
print(str((int(x.split('-')[0])+int(x.split('-')[1]))/2))
