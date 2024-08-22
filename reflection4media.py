# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 22:55:03 2023

@author: Noah Lee
"""
import numpy as np

def reflection4media(lambda_val, n0, k0, n1, k1, d1, n2, k2, d2, n3, k3):
    # According to the formula in Making Graphene Visible
    # EXAMPLE:
    # Air / MoS2 / SiO2 / Si
    # Medium 0 = air (semi-infinite)
    # Medium 1 = MoS2 (thickness d1)
    # Medium 2 = SiO2 (thickness d2)
    # Medium 3 = Si (semi-infinite)
    # nni = complex refractive index of medium 'i'
    nn0 = np.array(n0 - 1j * k0)
    nn1 = np.array(n1 - 1j * k1)
    nn2 = np.array(n2 - 1j * k2)
    nn3 = np.array(n3 - 1j * k3)
    # rii+1 = Fresnel coefficient at the interface between the media 'i' and 'i+1'
    r01 = (nn0 - nn1) / (nn0 + nn1)
    r12 = (nn1 - nn2) / (nn1 + nn2)
    r23 = (nn2 - nn3) / (nn2 + nn3)
    # phii = phase shift induced by the propagation of the light beam in mean 'i'
    phi1 = 2 * np.pi * nn1 * d1 / lambda_val
    phi2 = 2 * np.pi * nn2 * d2 / lambda_val
    # calculation of the reflection coefficient
    r_num = (r01 * np.exp(1j * (phi1 + phi2)) + r12 * np.exp(-1j * (phi1 - phi2)) + r23 * np.exp(-1j * (phi1 + phi2)) + r01 * r12 * r23 * np.exp(1j * (phi1 - phi2)))
    r_den = (np.exp(1j * (phi1 + phi2)) + r01 * r12 * np.exp(-1j * (phi1 - phi2)) + r01 * r23 * np.exp(-1j * (phi1 + phi2)) + r12 * r23 * np.exp(1j * (phi1 - phi2)))
    r = (r_num / r_den)
    R = np.conj(r) * r
    R = abs(R)
    return R

