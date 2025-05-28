#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install mmh3')


# In[ ]:


import mmh3  


player_id = "123456"

hash_value = mmh3.hash(player_id)


total_range = 100
group_a_range = int(total_range * 0.5)
group_b_range = group_a_range + int(total_range * 0.25)


if hash_value % total_range < group_a_range:
    group = "A"
elif hash_value % total_range < group_b_range:
    group = "B"
else:
    group = "C"

print(f" {player_id}  {group}")


# In[ ]:


import random
import string
import mmh3


def generate_random_id(length=8):
    all_characters = string.ascii_letters + string.digits
    return ''.join(random.choice(all_characters) for i in range(length))


random_ids = [generate_random_id() for _ in range(100)]


total_range = 100
group_a_range = int(total_range * 0.5)
group_b_range = group_a_range + int(total_range * 0.25)


for player_id in random_ids:
    
    hash_value = mmh3.hash(player_id)

    
    if hash_value % total_range < group_a_range:
        group = "A"
    elif hash_value % total_range < group_b_range:
        group = "B"
    else:
        group = "C"

    print(f" {player_id}  {group}")


# In[ ]:


# Change the Font to Chinese because there is Chinese Font in matplotlib

get_ipython().system('wget -O TaipeiSansTCBeta-Regular.ttf https://drive.google.com/uc?id=1eGAsTN1HBpJAkeVM57_C7ccp7hbgSz3_&export=download')

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager

fontManager.addfont('TaipeiSansTCBeta-Regular.ttf')
mpl.rc('font', family='Taipei Sans TC Beta')


# In[ ]:


import random
import string
import mmh3
from scipy.stats import chi2_contingency


def generate_random_id(length=8):
    all_characters = string.ascii_letters + string.digits
    return ''.join(random.choice(all_characters) for i in range(length))


random_ids = [generate_random_id() for _ in range(10000)]


total_range = 100
group_a_range = int(total_range * 0.5)
group_b_range = group_a_range + int(total_range * 0.25)


group_a_count = 0
group_b_count = 0
group_c_count = 0


for player_id in random_ids:
    
    hash_value = mmh3.hash(player_id)

    
    if hash_value % total_range < group_a_range:
        group_a_count += 1
    elif hash_value % total_range < group_b_range:
        group_b_count += 1
    else:
        group_c_count += 1


total_count = len(random_ids)
actual_ratio_a = group_a_count / total_count
actual_ratio_b = group_b_count / total_count
actual_ratio_c = group_c_count / total_count


print(f" A : {actual_ratio_a * 100:.2f}%")
print(f" B : {actual_ratio_b * 100:.2f}%")
print(f" C : {actual_ratio_c * 100:.2f}%")


observed = [group_a_count, group_b_count, group_c_count]
expected = [total_count * 0.5, total_count * 0.25, total_count * 0.25]
chi2, p, _, _ = chi2_contingency([observed, expected])


print(f": {chi2:.2f}")
print(f"p : {p:.4f}")
if p > 0.05:
    print("，。")
else:
    print("，。")


import matplotlib.pyplot as plt

labels = [' A', ' B', ' C']
sizes = [actual_ratio_a, actual_ratio_b, actual_ratio_c]
colors = ['lightcoral', 'lightskyblue', 'lightgreen']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%', startangle=140)
plt.axis('equal')
plt.title('')
plt.show()


# In[ ]:




