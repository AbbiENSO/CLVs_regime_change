#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 08:26:23 2023

@author: alessandro
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
import scipy as scipy
import os
import seaborn as sns
import sys

os.chdir('/home/alessandro/Desktop/PhD/qgs')

# %% functions


def rimuovi_neg(array):
    array_senza_zeri = [x for x in array if x >= 1]
    return array_senza_zeri



def filt(array):
    array_senza_zeri = [x for x in array if x <= 15]
    return array_senza_zeri


def convert_to_square_wave(peaks, minimum, maximum):
    square_wave = []
    for i in range(len(peaks) - 1):
        time_diff = peaks[i+1] - peaks[i]
        if minimum <= time_diff <= maximum:
            square_wave += [0] * time_diff
        else:
            square_wave += [1] * time_diff
    return square_wave

# Function to plot CDFs and perform KS test


def plot_cdf_and_ks(ax, data1, data2, label1, label2, xlabel):
    sorted_data1 = np.sort(data1)
    sorted_data2 = np.sort(data2)
    n1 = len(sorted_data1)
    n2 = len(sorted_data2)
    y1 = np.arange(1, n1 + 1) / n1
    y2 = np.arange(1, n2 + 1) / n2
    ax.plot(sorted_data1, y1, label=label1, color="blue")
    ax.plot(sorted_data2, y2, label=label2, color="red")
    ax.set_xlabel(xlabel)
    ax.set_ylabel('CDF')
    ks_statistic, ks_p_value = scipy.stats.ks_2samp(data1, data2)
    ax.text(0.1, 0.9, f'KS Statistic: {ks_statistic:.2f}',
            transform=ax.transAxes, fontsize=10)
    ax.text(0.1, 0.8, f'p-value: {ks_p_value:.4f}',
            transform=ax.transAxes, fontsize=10)
    ax.legend(loc='lower right')


def stats(array):
    # Calcola il momento primo
    momento_primo = np.mean(array)

    # Calcola il momento secondo
    momento_secondo = np.var(array)

    # Calcola la skewness
    skewness = np.mean(((array - momento_primo) / np.std(array))**3)

    # Calcola la kurtosis
    kurtosis = np.mean(((array - momento_primo) / np.std(array))**4) - 3

    # Calcola il valore massimo e il valore minimo
    valore_massimo = np.max(array)
    valore_minimo = np.min(array)

    # Stampa una tabella in formato LaTeX
    print(r"\begin{tabular}{|c|c|}")
    print(r"\hline")
    print(r"Statistiche & Valore \\")
    print(r"\hline")
    print(r"Momento Primo & {:.4f} \\".format(momento_primo))
    print(r"Momento Secondo & {:.4f} \\".format(momento_secondo))
    print(r"Skewness & {:.4f} \\".format(skewness))
    print(r"Kurtosis & {:.4f} \\".format(kurtosis))
    print(r"Valore Massimo & {:.4f} \\".format(valore_massimo))
    print(r"Valore Minimo & {:.4f} \\".format(valore_minimo))
    print(r"\hline")
    print(r"\end{tabular}")



# %% IMPORT FILES
ctraj = pd.read_csv("ctraj_ag1662.0.csv")
ctraj.pop(ctraj.columns[0])
print(ctraj)

angles = pd.read_csv("angles_ag166.2.csv")
angles.pop(angles.columns[0])
print(angles)

a12 = np.array(angles["e1e2"])
a23 = np.array(angles["e2e3"])
a13 = np.array(angles["e1e3"])
ts = np.array(ctraj["y"])
# %% PLot TS

t0 = 0
tw = 10000.
t = 30000.
dt = 0.01
mdt = 0.01
nt = int((t-tw)/dt)
t_plot = np.linspace(tw, t, nt+1)

tin = 200
tfin = 5000
rhov = 166.2


# %% ts
# # Carica i dati o definisci a12, a23, a13, ts e t_plot

# # Creazione della figura con quattro subplot
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

# # Tracciamento del coseno $\cos(\\alpha)$ nel primo subplot
# ax1.plot(t_plot[tin:tfin], a12[tin:tfin], color='red', label="$\cos(\\alpha)$")
# ax1.set_ylabel("$\cos(\\alpha)$")
# ax1.grid(True, linestyle='--', color='gray')

# # Tracciamento del coseno $\cos(\\beta)$ nel secondo subplot
# ax2.plot(t_plot[tin:tfin], a23[tin:tfin], color='red', label="$\cos(\\beta)$")
# ax2.set_ylabel("$\cos(\\beta)$")
# ax2.grid(True, linestyle='--', color='gray')

# # Tracciamento del coseno $\cos(\\gamma)$ nel terzo subplot
# ax3.plot(t_plot[tin:tfin], a13[tin:tfin], color='red', label="$\cos(\\gamma)$")
# ax3.set_ylabel("$\cos(\\gamma)$")
# ax3.grid(True, linestyle='--', color='gray')

# # Tracciamento della traiettoria blu nel quarto subplot
# ax4.plot(t_plot[tin:tfin], ts[tin:tfin] / max(ts[tin:tfin]), color='blue', label='$y(t)$')
# ax4.set_xlabel('t')
# ax4.set_ylabel("$y(t)$")
# ax4.grid(True, linestyle='--', color='gray')


# plt.tight_layout()
# plt.show()

# %% rolling shit

tin = 200
tfin = 5000

# Carica i dati o definisci a12, a23, a13, ts
plt.rcParams.update({'font.size': 13})
# Carica i dati o definisci a12, a23, a13, ts e rolling_mean per le altre curve "a" mediate
sns.set_style('darkgrid')
window_size = 200

# Calcola le curve mediate per a12, a23 e a13
rolling_mean_a12 = angles["e1e2"].rolling(window=window_size).mean()
rolling_mean_a12 = np.array(rolling_mean_a12)

rolling_mean_a23 = angles["e2e3"].rolling(window=window_size).mean()
rolling_mean_a23 = np.array(rolling_mean_a23)

rolling_mean_a13 = angles["e1e3"].rolling(window=window_size).mean()
rolling_mean_a13 = np.array(rolling_mean_a13)

# # Definisci i tempi per le linee verticali
# vertical_lines = [10029.5, 10031, 10045, 10050]

# Creazione della figura con tre subplot, un asse x comune e una traiettoria blu
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Tracciamento della traiettoria blu su tutti e tre i subplot
ax1.plot(t_plot[tin:tfin], ts[tin:tfin] / max(ts[tin:tfin]),
         label='$y(t)$', color='blue', alpha=0.5)
ax2.plot(t_plot[tin:tfin], ts[tin:tfin] /
         max(ts[tin:tfin]), color='blue', alpha=0.5)
ax3.plot(t_plot[tin:tfin], ts[tin:tfin] /
         max(ts[tin:tfin]), color='blue', alpha=0.5)

# Tracciamento delle curve mediate in rosso su ciascun subplot
ax1.plot(t_plot[tin:tfin], rolling_mean_a12[tin:tfin]**2,
         alpha=0.8, color="red", label="$\cos(\\alpha)$")
#ax1.plot(t_plot[tin:tfin], heav[tin:tfin],alpha=0.8, color="red", label="$\cos(\\alpha)$")
ax2.plot(t_plot[tin:tfin], rolling_mean_a23[tin:tfin],
         alpha=0.8, color="red", label="$\cos(\\beta)$")
ax3.plot(t_plot[tin:tfin], rolling_mean_a13[tin:tfin],
         alpha=0.8, color="red", label="$\cos(\\gamma)$")

# # Aggiungi le linee verticali ai tempi specificati
# for t in vertical_lines:
#     ax1.axvline(t, color='black', linestyle='--', linewidth=1.6)
#     ax2.axvline(t, color='black', linestyle='--', linewidth=1.6)
#     ax3.axvline(t, color='black', linestyle='--', linewidth=1.6)

# Personalizzazione degli assi e delle legende
ax1.set_ylabel('$y(t)$')
ax1.legend()
ax1.grid(True, linestyle='--', color='gray')

ax2.set_ylabel('$y(t)$')
ax2.legend()
ax2.grid(True, linestyle='--', color='gray')

ax3.set_xlabel('t')
ax3.set_ylabel('$y(t)$')
ax3.legend()
ax3.grid(True, linestyle='--', color='gray')

# Rimuovi l'etichetta dell'asse x dal subplot superiore
ax1.set_xlabel('')


plt.show()


# %% find peaks in ts:

deltax = (1.13+1.14)/2
alpha = 9.5
beta = 10.2

peaks, _ = find_peaks(ts, height=(alpha, beta))
lam_len = np.ediff1d(peaks)
lamlen = np.zeros(len(lam_len))

k = 0
for i in range(0, len(lam_len)):
    if lam_len[i] < 1.15 and lam_len[i] > 0:
        k = k + 1
        # print(k)
    else:
        lamlen[i] = (k - 1)*deltax
        # print(k-1)
        k = 0

lamlen = rimuovi_neg(lamlen)

step = np.zeros(len(lam_len))

# Utilizza la funzione con i picchi e l'intervallo desiderato

minn = 112
maxx = 115
square_wave = convert_to_square_wave(peaks, minn, maxx)

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(t_plot[tin:tfin], ts[tin:tfin] / max(ts[tin:tfin]),
        label='$y(t)$', color='blue', alpha=0.5)
ax.plot(t_plot[tin:tfin], square_wave[tin:tfin],
        alpha=0.8, color="red", label="$\cos(\\alpha)$")

plt.show()

# %% define a stepfunction for clvs

zeros = np.zeros(len(rolling_mean_a12))
heav = np.zeros(len(rolling_mean_a12))

for j in range(len(rolling_mean_a12)):
    if 0.72 <= rolling_mean_a12[j]**2 <= 1:
        heav[j] = 0
    else:
        heav[j] = 1

# %% Plot clvs stat1

tin = 200
tfin = 5000

lag = tin +240
lag2 = tfin +240
fig, (ax, ax1, bx,kx,cx) = plt.subplots(5,1,figsize=(10, 10),sharex=True)

ax.plot(t_plot[tin:tfin], ts[tin:tfin] / max(ts[tin:tfin]),label='$y(t)$', color='blue', alpha=0.5)
ax.plot(t_plot[tin:tfin], rolling_mean_a12[tin:tfin]**2,alpha=0.8, color="red", label="$\cos(\\alpha)$")
ax.set_ylabel("$\cos(\\gamma)$")
ax.legend()

ax1.plot(t_plot[tin:tfin], ts[tin:tfin] / max(ts[tin:tfin]),label='$y(t)$', color='blue', alpha=0.5)
ax1.plot(t_plot[tin:tfin], heav[tin:tfin],alpha=0.8, color="red", label="sqr CLVs")
ax1.set_ylabel("$CLV sqr")
ax1.legend()

bx.plot(t_plot[tin:tfin], rolling_mean_a12[tin:tfin]**2,label='$\cos(\\alpha)$', color='blue', alpha=0.5)
bx.plot(t_plot[tin:tfin], heav[tin:tfin],alpha=0.8, color="red", label="sqr CLV")
bx.set_ylabel("CLV and sqr")
bx.legend()

kx.plot(t_plot[tin:tfin], square_wave[tin:tfin],label='sqr y', color='green', alpha=1)
kx.plot(t_plot[tin:tfin], ts[tin:tfin]/max(ts[tin:tfin]),alpha=0.5, color="blue", label="$y(t)$")
kx.set_ylabel("TS and TS sqr")
kx.legend()

cx.plot(t_plot[lag:lag2], square_wave[tin:tfin],label='y', color='purple', alpha=0.8)
cx.plot(t_plot[tin:tfin], heav[tin:tfin],alpha=0.8, color="red", label="CLV")
cx.set_ylabel("CLVs vs TS")
cx.legend()

cx.set_xlabel('t')


x_limit_start = 10005
x_limit_end = 10050

ax.set_xlim(x_limit_start, x_limit_end)
ax1.set_xlim(x_limit_start, x_limit_end)
bx.set_xlim(x_limit_start, x_limit_end)
cx.set_xlim(x_limit_start, x_limit_end)
kx.set_xlim(x_limit_start, x_limit_end)

plt.show()


# %% hit or miss:

# considero heav come forecast
# considero squere come dato

# Hit:             heav = 0, sqr = 0
# correct rej:     heva = 1, sqr = 1
# miss             heav = 1, sqr = 0
# False alarm:     heav = 0, sqr = 1

hit = 0
miss = 0
false_alarm = 0
correct_rej = 0

asd = len(square_wave) - 100 + 22

a = square_wave[0:asd]
b = heav[230:-1]
c_t = scipy.stats.pearsonr(a, b)

len(a)
len(b)

for j in range(len(a)):
    if a[j] == 1 and b[j] == 1:
        hit += 1
    elif a[j] == 0 and b[j] == 0:
        correct_rej += 1
    elif a[j] == 0 and b[j] == 1:
        false_alarm += 1
    elif a[j] == 1 and b[j] == 0:
        miss += 1

fail = false_alarm + miss
success = hit + correct_rej

per_fail = (fail/len(a))*100
per_success = (success/len(a))*100

per_hit = (hit/success)*100
per_CR = (correct_rej/success)*100
per_miss = (miss/fail)*100
per_FA = (false_alarm/fail)*100


# %% Single time series
tin = 0
tfin = 5000
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(t_plot[tin:tfin], a[tin:tfin], label='$y(t)$', color='blue', alpha=0.5)
ax.plot(t_plot[tin:tfin], b[tin:tfin], alpha=0.8,
        color="red", label="$\cos(\\alpha)$")
ax.set_ylabel("$\cos(\\gamma)$")

plt.show()


# %% Compute Laminar lenghts

# a è dai dati
# b è dagli angoli

a = square_wave
b = heav
ll = []
ll1 = []
k = 0



for j in range(len(a)):
    if a[j] == 0:
        k += 1
    else:
        ll.append(k*dt)
        k = 0
k = 0
for i in range(len(b)):
    if b[i] == 0:
        k += 1
    else:
        ll1.append(k*dt)
        k = 0

ll = ll[1:-1]
ll1 = ll1[1:-1]

ll = rimuovi_neg(ll)

ll1 = rimuovi_neg(ll1)
ll1 = filt(ll1)

# %% Statistics

WD = scipy.stats.wasserstein_distance(ll, ll1)  # Wessestrain distance
res = scipy.stats.mannwhitneyu(ll, ll1)   # Non-Par t-test


print("\n")
print("\t                      HIT OR MISS TEST & CORRELATION")
print("___________________________________________________________________________________")
print("Success \t \t Fail \t \t Correlation")
print(round(per_success, 2), "\t \t", round(per_fail, 2), "\t \t", c_t, "\n")

print("Hit \t \t CR \t \t Miss \t FA ")
print(round(per_hit, 2), "\t", round(per_CR, 2), "\t",
      round(per_miss, 2), "\t", round(per_FA, 2))
print("___________________________________________________________________________________")

###############################################################################
print("\n")
print("\t                           NON PARAMETRIC TESTS")
print("___________________________________________________________________________________")

print("WD:\t", WD, "\n")
print("MANN:\t",res)
print("___________________________________________________________________________________\n")
#
##############################################################################

print("\n")
print("\t          Latex table with statiatics")
print("_____________________________________________________")
print("_____________________________________________________")
stats(ll)
print("\n")
stats(ll1)
print("\n")

original_stdout = sys.stdout # Save a reference to the original standard output
with open('Stat_result_r=().txt', 'w') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print("\n")
    print("\t                      HIT OR MISS TEST & CORRELATION")
    print("___________________________________________________________________________________")
    print("Success \t \t Fail \t \t Correlation")
    print(round(per_success, 2), "\t \t", round(per_fail, 2), "\t \t", c_t, "\n")

    print("Hit \t \t CR \t \t Miss \t FA ")
    print(round(per_hit, 2), "\t", round(per_CR, 2), "\t",
          round(per_miss, 2), "\t", round(per_FA, 2))
    print("___________________________________________________________________________________")

    ###############################################################################
    print("\n")
    print("\t                           NON PARAMETRIC TESTS")
    print("___________________________________________________________________________________")

    print("WD:\t", WD, "\n")
    print("MANN:\t",res)
    print("___________________________________________________________________________________\n")
    #
    ##############################################################################

    print("\n")
    print("\t          Latex table with statiatics")
    print("_____________________________________________________")
    print("_____________________________________________________")
    stats(ll)
    print("\n")
    stats(ll1)
    print("\n")
    sys.stdout = original_stdout # Reset the standard output to its original value
###############################################################################

# %% Histograms, correlarion and pie chart

# %% Pie chart
plt.rcParams.update({'font.size': 8})

# Make data: I have 3 groups and 7 subgroups
group_names = ['Success', 'Fail']
group_size = [90.76, 9.24]
subgroup_names = ['Hit', 'C.R.', 'Miss', 'F.A.']
subgroup_size = [39.13, 51.63, 3.11, 6.13]

# Create colors
a1, b1 = [plt.cm.Greens, plt.cm.Reds]

sns.reset_defaults()
sns.set_style('darkgrid')
num_bins = 11

fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("")

ax.hist(ll, alpha=0.5, label='From data', density=True,
        bins=num_bins, color="blue", edgecolor='none')
ax.hist(ll1, alpha=0.5, label='From $<CLV_1|CLV_2>$', density=True,
        bins=num_bins, color="red", edgecolor='none')
ax.legend(loc='upper left')
ax.set_ylabel("Density")
ax.set_title('W.D. = {WD}'.format(WD=round(WD,2)))
ax.set_xlabel("Laminar Length")


bx.axis('equal')
mypie, _ = bx.pie(group_size, radius=1.3, labels=group_names,
                  colors=[a1(0.6), b1(0.6)])
plt.setp(mypie, width=0.3, edgecolor='black')

# Second Ring (Inside)
mypie2, _ = bx.pie(subgroup_size, radius=1.3-0.3, labels=subgroup_names,
                   labeldistance=0.7, colors=[a1(0.5), a1(0.3), b1(0.2), b1(0.4)])
plt.setp(mypie2, width=0.4, edgecolor='black')
plt.margins(0, 0)

plt.show()


# %% Boxplot

data = [ll, ll1]

# Creare un boxplot con notches
sns.set(style="darkgrid")
plt.figure(figsize=(6, 4))
sns.boxplot(data=data, width=0.5, notch=True)

# Aggiungi etichette
plt.xticks([0, 1], ['From data', 'From CLVs'])
plt.xlabel("Groups")
plt.ylabel("Value")
plt.title("Boxplot Of Laminar Lengths")

# Mostra il grafico
plt.show()
