import numpy as np

def to_list(data,index_list):
    return [data[:,index_list[i]:index_list[i+1]] for i in range(len(index_list)-1)]

N_list_Skyrme=np.array([1084,1901,2459,4054])
index_list_Skyrme=[N_list_Skyrme[:i].sum() for i in range(5)]

# FchCa FchPb FwCa FwPb RchCa(fm) RchZr(fm) RchPb(fm) BECa(MeV) BEZr(MeV) BEPb(MeV) 
# Rn_Pb208 (fm) Rp_Pb208(fm) Rn_Ca48(fm) Rp_Ca48(fm)
PrexCrex_Skyrme=np.loadtxt('./data/PrexCrex_Skyrme.txt').transpose()
PrexCrex_Skyrme_list=to_list(PrexCrex_Skyrme,index_list_Skyrme)

# t0 (MeVfm^3) t1(MeVfm^5) t2(MeVfm^5) t3(MeVfm^3) x0() x1() x2() x3() alpha() b4(fm^4) b4p(fm^4)
eos_args_Skyrme =np.loadtxt('./data/eos_args_Skyrme.txt').transpose()
eos_args_Skyrme_list=to_list(eos_args_Skyrme,index_list_Skyrme)

# [m_eff_SNM,ns,BE,J_SYM,L_SYM,K_SNM,K_SYM]
SAT_Skyrme =np.loadtxt('./data/SAT_Skyrme.txt').transpose()
SAT_Skyrme_list=to_list(SAT_Skyrme,index_list_Skyrme)

# [m_eff_n_PNM,m_eff_p_PNM,ns,J_PNM,L_PNM,K_SNM,K_SYM]
SAT_PNM_Skyrme =np.loadtxt('./data/SAT_PNM_Skyrme.txt').transpose()
SAT_PNM_Skyrme_list=to_list(SAT_PNM_Skyrme,index_list_Skyrme)


BulK_J_Skyrme=np.loadtxt('./data/BulK_J_Skyrme.txt').transpose()
BulK_J_Skyrme_list=to_list(BulK_J_Skyrme,index_list_Skyrme)

BulK_K_Skyrme=np.loadtxt('./data/BulK_K_Skyrme.txt').transpose()
BulK_K_Skyrme_list=to_list(BulK_K_Skyrme,index_list_Skyrme)

BulK_L_Skyrme=np.loadtxt('./data/BulK_L_Skyrme.txt').transpose()
BulK_L_Skyrme_list=to_list(BulK_L_Skyrme,index_list_Skyrme)

BulK_P_Skyrme=np.loadtxt('./data/BulK_P_Skyrme.txt').transpose()
BulK_P_Skyrme_list=to_list(BulK_P_Skyrme,index_list_Skyrme)


N_list_RMF=np.loadtxt('./data/N_list_RMF.txt').astype('int')
N_list_RMF_to_Skyrme=np.loadtxt('./data/N_list_RMF_to_Skyrme.txt').astype('int')
index_list_RMF=[N_list_RMF[:i].sum() for i in range(5)]
index_list_RMF_to_Skyrme=[N_list_RMF_to_Skyrme[:i].sum() for i in range(5)]

# FchCa FchPb FwCa FwPb RchCa(fm) RchZr(fm) RchPb(fm) BECa(MeV) BEZr(MeV) BEPb(MeV) 
# Rn_Pb208 (fm) Rp_Pb208(fm) Rn_Ca48(fm) Rp_Ca48(fm)
PrexCrex_RMF=np.loadtxt('./data/PrexCrex_RMF.txt').transpose()
PrexCrex_RMF_list=to_list(PrexCrex_RMF,index_list_RMF)
PrexCrex_RMF_to_Skyrme=np.loadtxt('./data/PrexCrex_RMF_to_Skyrme.txt').transpose()
PrexCrex_RMF_to_Skyrme_list=to_list(PrexCrex_RMF_to_Skyrme,index_list_RMF_to_Skyrme)

# m_sigma (MeV) g^2_sigma() g^2_delta() g^2_omega() g^2_rho() kappa(MeV) lambda() Lambda() zeta()
eos_args_RMF =np.loadtxt('./data/eos_args_RMF.txt').transpose()
eos_args_RMF_list=to_list(eos_args_RMF,index_list_RMF)
eos_args_RMF_to_Skyrme =np.loadtxt('./data/eos_args_RMF_to_Skyrme.txt').transpose()
eos_args_RMF_to_Skyrme_list=to_list(eos_args_RMF_to_Skyrme,index_list_RMF_to_Skyrme)

# [m_eff_SNM,ns,BE,J_SYM,L_SYM,K_SNM,K_SYM]
SAT_RMF =np.loadtxt('./data/SAT_RMF.txt').transpose()
SAT_RMF_list=to_list(SAT_RMF,index_list_RMF)
SAT_RMF_to_Skyrme =np.loadtxt('./data/SAT_RMF_to_Skyrme.txt').transpose()
SAT_RMF_to_Skyrme_list=to_list(SAT_RMF_to_Skyrme,index_list_RMF_to_Skyrme)

# [m_eff_n_PNM,m_eff_p_PNM,ns,J_PNM,L_PNM,K_SNM,K_SYM]
SAT_PNM_RMF =np.loadtxt('./data/SAT_PNM_RMF.txt').transpose()
SAT_PNM_RMF_list=to_list(SAT_PNM_RMF,index_list_RMF)
SAT_PNM_RMF_to_Skyrme =np.loadtxt('./data/SAT_PNM_RMF_to_Skyrme.txt').transpose()
SAT_PNM_RMF_to_Skyrme_list=to_list(SAT_PNM_RMF_to_Skyrme,index_list_RMF_to_Skyrme)


BulK_J_RMF=np.loadtxt('./data/BulK_J_RMF.txt').transpose()
BulK_J_RMF_list=to_list(BulK_J_RMF,index_list_RMF)

BulK_K_RMF=np.loadtxt('./data/BulK_K_RMF.txt').transpose()
BulK_K_RMF_list=to_list(BulK_K_RMF,index_list_RMF)

BulK_L_RMF=np.loadtxt('./data/BulK_L_RMF.txt').transpose()
BulK_L_RMF_list=to_list(BulK_L_RMF,index_list_RMF)

BulK_E_RMF=np.loadtxt('./data/BulK_E_RMF.txt').transpose()
BulK_E_RMF_list=to_list(BulK_E_RMF,index_list_RMF)

BulK_P_RMF=np.loadtxt('./data/BulK_P_RMF.txt').transpose()
BulK_P_RMF_list=to_list(BulK_P_RMF,index_list_RMF)




# PrexCrex_RMF =np.loadtxt('./data/PrexCrex_RMF.txt').transpose()

# BulK_J_RMF=np.loadtxt('./data/BulK_J_RMF.txt').transpose()
# BulK_K_RMF=np.loadtxt('./data/BulK_K_RMF.txt').transpose()
# BulK_L_RMF=np.loadtxt('./data/BulK_L_RMF.txt').transpose()
# BulK_P_RMF=np.loadtxt('./data/BulK_P_RMF.txt').transpose()
# BulK_E_RMF=np.loadtxt('./data/BulK_E_RMF.txt').transpose()

# BulK_J_PNM_RMF=np.loadtxt('./data/BulK_J_PNM_RMF.txt').transpose()
# BulK_K_PNM_RMF=np.loadtxt('./data/BulK_K_PNM_RMF.txt').transpose()
# BulK_L_PNM_RMF=np.loadtxt('./data/BulK_L_PNM_RMF.txt').transpose()

# SAT_RMF=np.loadtxt('./data/SAT_RMF.txt').transpose()
# SAT_PNM_RMF=np.loadtxt('./data/SAT_PNM_RMF.txt').transpose()
# eos_args_RMF=np.loadtxt('./data/eos_args_RMF.txt').transpose()
# eos_args_sat_RMF=np.loadtxt('./data/eos_args_sat_RMF.txt').transpose()

# FchFwCa=PrexCrex[0]-PrexCrex[2]
# FchFwPb=PrexCrex[1]-PrexCrex[3]
# RpRnCa =PrexCrex[12]-PrexCrex[13]
# RpRnPb =PrexCrex[10]-PrexCrex[11]
# BECaZrPb =PrexCrex[7:10]
# FchFwCa_RMF=PrexCrex_RMF[0]-PrexCrex_RMF[2]
# FchFwPb_RMF=PrexCrex_RMF[1]-PrexCrex_RMF[3]
# RpRnCa_RMF =PrexCrex_RMF[12]-PrexCrex_RMF[13]
# RpRnPb_RMF =PrexCrex_RMF[10]-PrexCrex_RMF[11]
# BECaZrPb_RMF =PrexCrex_RMF[7:10]