import os
import numpy
import pandas as pd
import scipy.stats as st

os.chdir('/Users/jarvis/Dropbox/Apps/HypertensionOutputs')


def summary_cost(int_details,ctrl_m,ctrl_f,trt_m,trt_f, text):
    int_dwc = 1 / (1 + discount_rate) ** numpy.array(range(time_horizon))
    int_c = numpy.array([[prog_cost] * time_horizon for i in range(1)])
    int_cost = numpy.sum(numpy.dot(int_c, int_dwc))

    female_pop = 188340000
    male_pop = 196604000

    pop = female_pop + male_pop

    f_prop = female_pop / pop
    m_prop = male_pop / pop

    samples = ctrl_m.shape[0]

    cs = 0
    nq = 0
    ic = [0.00 for i in range(samples)]
    q_gained = [0.00 for i in range(samples)]
    q_inc_percent = [0.00 for i in range(samples)]
    htn_cost = [0.00 for i in range(samples)]
    cvd_cost = [0.00 for i in range(samples)]
    net_cost = [0.00 for i in range(samples)]
    exp_inc_per = [0.00 for i in range(samples)]

    for i in range(samples):
        q_gained[i] = (((ctrl_m.loc[i, "Average DALYs"] - trt_m.loc[i, "Average DALYs"])* m_prop) + ((ctrl_f.loc[i, "Average DALYs"] - trt_f.loc[i, "Average DALYs"])* f_prop))
        q_inc_percent[i] = q_gained[i] * 100/((ctrl_m.loc[i, "Average DALYs"] * m_prop) + (ctrl_f.loc[i, "Average DALYs"] *f_prop))
        htn_cost[i] = int_cost + ((trt_m.loc[i, "Average HTN Cost"] - ctrl_m.loc[i, "Average HTN Cost"]) * m_prop) + ((trt_f.loc[i, "Average HTN Cost"] - ctrl_f.loc[i, "Average HTN Cost"]) * f_prop)
        cvd_cost[i] = ((trt_m.loc[i, "Average CVD Cost"] - ctrl_m.loc[i, "Average CVD Cost"] + trt_m.loc[i, "Average Chronic Cost"] - ctrl_m.loc[i, "Average Chronic Cost"]) * m_prop) + ((trt_f.loc[i, "Average CVD Cost"] - ctrl_f.loc[i, "Average CVD Cost"] + trt_f.loc[i, "Average Chronic Cost"] - ctrl_f.loc[i, "Average Chronic Cost"]) * f_prop)
        exp_inc_per[i] = (((trt_m.loc[i, "Average Cost"] - ctrl_m.loc[i, "Average Cost"] + int_cost) * m_prop) + ((trt_f.loc[i, "Average Cost"] - ctrl_f.loc[i, "Average Cost"] + int_cost) * f_prop)) * 100 / ((ctrl_m.loc[i, "Average Cost"] * m_prop ) + (ctrl_f.loc[i, "Average Cost"] * f_prop))
        net_cost[i] = htn_cost[i] + cvd_cost[i]
        ic[i] = net_cost[i] / q_gained[i]

        if net_cost[i] < 0:
            cs = cs + 1
        if q_gained[i] < 0:
            nq = nq + 1

    budget_impact = numpy.mean(net_cost) * pop / time_horizon
    htn_percap = numpy.mean(htn_cost) / time_horizon
    cvd_percap = numpy.mean(cvd_cost) / time_horizon
    htn_annual = numpy.mean(htn_cost) * pop / time_horizon
    cvd_annual = numpy.mean(cvd_cost) * pop / time_horizon
    cost_inc = numpy.mean(exp_inc_per)
    ICER = numpy.mean(ic)
    QALY = numpy.mean(q_inc_percent)
    HTN = numpy.mean(htn_cost)
    CVD = numpy.mean(cvd_cost)
    icer_95 = st.t.interval(0.95, samples - 1, loc=numpy.mean(ic), scale=st.sem(ic))
    qaly_95 = st.t.interval(0.95, samples - 1, loc=numpy.mean(q_inc_percent), scale=st.sem(q_inc_percent))
    htn = st.t.interval(0.95, samples - 1, loc=numpy.mean(htn_cost), scale=st.sem(htn_cost))
    cvd = st.t.interval(0.95, samples - 1, loc=numpy.mean(cvd_cost), scale=st.sem(cvd_cost))
    cost_inc_95 = st.t.interval(0.95, samples - 1, loc=numpy.mean(exp_inc_per), scale=st.sem(exp_inc_per))

    if budget_impact < 0:
        m_icer = 'Cost Saving'
        s_icer = 'CS'
    else:
        m_icer = numpy.mean(net_cost) / numpy.mean(q_gained)
        s_icer = str(numpy.round(m_icer,1))

    m_daly = str(numpy.round(QALY,3)) + "\n(" + str(numpy.round(qaly_95[0],3)) + " to " + str(numpy.round(qaly_95[1],3)) + ")"
    m_htn = str(numpy.round(HTN,2)) + "\n(" + str(numpy.round(htn[0],2)) + " to " + str(numpy.round(htn[1],2)) + ")"
    m_cvd = str(numpy.round(CVD,2)) + "\n(" + str(numpy.round(cvd[0],2)) + " to " + str(numpy.round(cvd[1],2)) + ")"
    m_costinc = str(numpy.round(cost_inc, 2)) + "\n(" + str(numpy.round(cost_inc_95[0], 2)) + " to " + str(numpy.round(cost_inc_95[1], 2)) + ")"
    m_budget = str(numpy.round(budget_impact,0)/1000)

    err_cost = 1.96 * st.sem(exp_inc_per)
    err_daly = 1.96 * st.sem(q_inc_percent)

    str_icer = text + " (" + s_icer + ")"

    detailed = [int_details[2], int_details[0], int_details[1], int_details[3], int_details[4], ICER, icer_95[0],icer_95[1], QALY, qaly_95[0], qaly_95[1], htn[0], htn[1], cvd[0], cvd[1], budget_impact, htn_annual, cvd_annual, htn_percap, cvd_percap, cs, nq]
    manuscript = [int_details[2], int_details[0], int_details[1], int_details[3], int_details[4], m_icer,  m_daly, m_costinc, m_htn, m_cvd, m_budget, cs]
    plot = [text, str_icer, cost_inc, QALY, err_cost, err_daly]

    return detailed, manuscript, plot

summary_output = []
appendix_output = []
plot_output = []

'''Analysis 0: Baseline'''
time_horizon = 20
prog_cost = 0.13
discount_rate = 0.03

os.chdir('/Users/jarvis/Dropbox/Apps/HypertensionOutputs/15Aug_AWS3')
fname = [0.4, 0.3, 0, 0.8, 0.6]
file_name_m = ("Aspire_Male_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
file_name_f = ("Aspire_Female_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
control_m = pd.read_csv(file_name_m)
control_f = pd.read_csv(file_name_f)

fname = [0.7, 0.7, 1, 0.8, 0.8]
file_name_m = ("Aspire_Male_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
file_name_f = ("Aspire_Female_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
treatment_f = pd.read_csv(file_name_f)
treatment_m = pd.read_csv(file_name_m)

res = summary_cost(fname, control_m, control_f, treatment_m, treatment_f,'Base Case')
summary_output.append(res[0])
appendix_output.append(res[1])
plot_output.append(res[2])

'''Analysis 1: Doubled Medication Cost'''

time_horizon = 20
prog_cost = 0.13
discount_rate = 0.03

os.chdir('/Users/jarvis/Dropbox/Apps/HypertensionOutputs/PSAFinal')
fname = [0.4, 0.3, 0, 0.8, 0.6, 2, 0, 20]
file_name_m = ("Aspire_Male_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) +"_CF_"+ str(fname[5]) + "_RR_"+ str(fname[6])  + "_TH_"+ str(fname[7])  + ".csv")
file_name_f = ("Aspire_Female_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) +"_CF_"+ str(fname[5]) + "_RR_"+ str(fname[6])  + "_TH_"+ str(fname[7])  + ".csv")
control_m = pd.read_csv(file_name_m)
control_f = pd.read_csv(file_name_f)

fname = [0.7, 0.7, 1, 0.8, 0.8, 2, 0, 20]
file_name_f = ("Aspire_Female_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
            fname[4]) +"_CF_"+ str(fname[5]) + "_RR_"+ str(fname[6])  + "_TH_"+ str(fname[7])  + ".csv")
file_name_m = ("Aspire_Male_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
            fname[4]) +"_CF_"+ str(fname[5]) + "_RR_"+ str(fname[6])  + "_TH_"+ str(fname[7])  + ".csv")
treatment_f = pd.read_csv(file_name_f)
treatment_m = pd.read_csv(file_name_m)

res = summary_cost(fname, control_m, control_f, treatment_m, treatment_f,'2X Medication Cost')
summary_output.append(res[0])
appendix_output.append(res[1])
plot_output.append(res[2])

'''Analysis 2: Increased Programmatic Cost'''
time_horizon = 20
prog_cost = 0.13*4
discount_rate = 0.03

os.chdir('/Users/jarvis/Dropbox/Apps/HypertensionOutputs/15Aug_AWS3')
fname = [0.4, 0.3, 0, 0.8, 0.6]
file_name_m = ("Aspire_Male_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
file_name_f = ("Aspire_Female_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
control_m = pd.read_csv(file_name_m)
control_f = pd.read_csv(file_name_f)

fname = [0.7, 0.7, 1, 0.8, 0.8]
file_name_m = ("Aspire_Male_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
file_name_f = ("Aspire_Female_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
treatment_f = pd.read_csv(file_name_f)
treatment_m = pd.read_csv(file_name_m)

res = summary_cost(fname, control_m, control_f, treatment_m, treatment_f,'4X Programmatic Cost')
summary_output.append(res[0])
appendix_output.append(res[1])
plot_output.append(res[2])

'''Analysis 3: 20% reduction in baseline CVD risk'''
time_horizon = 20
prog_cost = 0.13
discount_rate = 0.03

os.chdir('/Users/jarvis/Dropbox/Apps/HypertensionOutputs/PSAFinal')
fname = [0.4, 0.3, 0, 0.8, 0.6, 1, 0.2, 20]
file_name_m = ("Aspire_Male_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) +"_CF_"+ str(fname[5]) + "_RR_"+ str(fname[6])  + "_TH_"+ str(fname[7])  + ".csv")
file_name_f = ("Aspire_Female_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) +"_CF_"+ str(fname[5]) + "_RR_"+ str(fname[6])  + "_TH_"+ str(fname[7])  + ".csv")
control_m = pd.read_csv(file_name_m)
control_f = pd.read_csv(file_name_f)

fname = [0.7, 0.7, 1, 0.8, 0.8, 1, 0.2, 20]
file_name_f = ("Aspire_Female_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
            fname[4]) +"_CF_"+ str(fname[5]) + "_RR_"+ str(fname[6])  + "_TH_"+ str(fname[7])  + ".csv")
file_name_m = ("Aspire_Male_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
            fname[4]) +"_CF_"+ str(fname[5]) + "_RR_"+ str(fname[6])  + "_TH_"+ str(fname[7])  + ".csv")
treatment_f = pd.read_csv(file_name_f)
treatment_m = pd.read_csv(file_name_m)

res = summary_cost(fname, control_m, control_f, treatment_m, treatment_f,'Reduced Baseline Risk')
summary_output.append(res[0])
appendix_output.append(res[1])
plot_output.append(res[2])

'''Analysis 4: NPCDCS Medication Protocol'''

os.chdir('/Users/jarvis/Dropbox/Apps/HypertensionOutputs/15Aug_AWS3')
fname = [0.4, 0.3, 0, 0.8, 0.6]
file_name_m = ("Aspire_Male_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
file_name_f = ("Aspire_Female_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
control_m = pd.read_csv(file_name_m)
control_f = pd.read_csv(file_name_f)

fname = [0.7, 0.7, 0, 0.8, 0.8]
file_name_m = ("Aspire_Male_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
file_name_f = ("Aspire_Female_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
treatment_m = pd.read_csv(file_name_m)
treatment_f = pd.read_csv(file_name_f)

res = summary_cost(fname, control_m, control_f, treatment_m, treatment_f,'NPCDCS Treatment Guideline')
summary_output.append(res[0])
appendix_output.append(res[1])
plot_output.append(res[2])

'''Analysis 5: Private Sector Cost'''

os.chdir('/Users/jarvis/Dropbox/Apps/HypertensionOutputs/PvtFinal')
fname = [0.4, 0.3, 0, 0.8, 0.6]
file_name_m = ("Aspire_Male_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
file_name_f = ("Aspire_Female_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
control_m = pd.read_csv(file_name_m)
control_f = pd.read_csv(file_name_f)

fname = [0.7, 0.7, 1, 0.8, 0.8]
file_name_m = ("Aspire_Male_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
file_name_f = ("Aspire_Female_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
treatment_m = pd.read_csv(file_name_m)
treatment_f = pd.read_csv(file_name_f)

res = summary_cost(fname, control_m, control_f, treatment_m, treatment_f,'Private Sector')
summary_output.append(res[0])
appendix_output.append(res[1])
plot_output.append(res[2])

'''Analysis 6: PubPvt Mix Cost'''

os.chdir('/Users/jarvis/Dropbox/Apps/HypertensionOutputs/PubPvtFinal')
fname = [0.4, 0.3, 0, 0.8, 0.6]
file_name_m = ("Aspire_Male_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
file_name_f = ("Aspire_Female_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
control_m = pd.read_csv(file_name_m)
control_f = pd.read_csv(file_name_f)

fname = [0.7, 0.7, 1, 0.8, 0.8]
file_name_m = ("Aspire_Male_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
file_name_f = ("Aspire_Female_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
treatment_m = pd.read_csv(file_name_m)
treatment_f = pd.read_csv(file_name_f)

res = summary_cost(fname, control_m, control_f, treatment_m, treatment_f,'Public-Private Mix')
summary_output.append(res[0])
appendix_output.append(res[1])
plot_output.append(res[2])

#Analysis 7: 10-year Time Horizon#
time_horizon = 10
prog_cost = 0.13
discount_rate = 0.03
os.chdir('/Users/jarvis/Dropbox/Apps/HypertensionOutputs/10yFinal')
fname = [0.4, 0.3, 0, 0.8, 0.6, 1, 0, 10]
file_name_m = ("Aspire_Male_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) +"_CF_"+ str(fname[5]) + "_RR_"+ str(fname[6])  + "_TH_"+ str(fname[7])  + ".csv")
file_name_f = ("Aspire_Female_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) +"_CF_"+ str(fname[5]) + "_RR_"+ str(fname[6])  + "_TH_"+ str(fname[7])  + ".csv")
control_m = pd.read_csv(file_name_m)
control_f = pd.read_csv(file_name_f)

fname = [0.7, 0.7, 1, 0.8, 0.8, 1, 0, 10]
file_name_f = ("Aspire_Female_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
            fname[4]) +"_CF_"+ str(fname[5]) + "_RR_"+ str(fname[6])  + "_TH_"+ str(fname[7])  + ".csv")
file_name_m = ("Aspire_Male_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
            fname[4]) +"_CF_"+ str(fname[5]) + "_RR_"+ str(fname[6])  + "_TH_"+ str(fname[7])  + ".csv")
treatment_f = pd.read_csv(file_name_f)
treatment_m = pd.read_csv(file_name_m)

res = summary_cost(fname, control_m, control_f, treatment_m, treatment_f,'10 year Horizon')
summary_output.append(res[0])
appendix_output.append(res[1])
plot_output.append(res[2])


'''Analysis 8: 40-year Time Horizon'''
time_horizon = 40
prog_cost = 0.13
discount_rate = 0.03
os.chdir('/Users/jarvis/Dropbox/Apps/HypertensionOutputs/40yFinal')
fname = [0.4, 0.3, 0, 0.8, 0.6, 1, 0, 40]
file_name_m = ("Aspire_Male_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) +"_CF_"+ str(fname[5]) + "_RR_"+ str(fname[6])  + "_TH_"+ str(fname[7])  + ".csv")
file_name_f = ("Aspire_Female_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) +"_CF_"+ str(fname[5]) + "_RR_"+ str(fname[6])  + "_TH_"+ str(fname[7])  + ".csv")
control_m = pd.read_csv(file_name_m)
control_f = pd.read_csv(file_name_f)

fname = [0.7, 0.7, 1, 0.8, 0.8, 1, 0, 40]
file_name_f = ("Aspire_Female_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
            fname[4]) +"_CF_"+ str(fname[5]) + "_RR_"+ str(fname[6])  + "_TH_"+ str(fname[7])  + ".csv")
file_name_m = ("Aspire_Male_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
            fname[4]) +"_CF_"+ str(fname[5]) + "_RR_"+ str(fname[6])  + "_TH_"+ str(fname[7])  + ".csv")
treatment_f = pd.read_csv(file_name_f)
treatment_m = pd.read_csv(file_name_m)

res = summary_cost(fname, control_m, control_f, treatment_m, treatment_f,'40 year Horizon')
summary_output.append(res[0])
appendix_output.append(res[1])
plot_output.append(res[2])

'''Analysis 9: Inlusion of Pill Disutility'''
time_horizon = 20
prog_cost = 0.13
discount_rate = 0.03

os.chdir('/Users/jarvis/Dropbox/Apps/HypertensionOutputs/PillDisFinal')
fname = [0.4, 0.3, 0, 0.8, 0.6]
file_name_m = ("Aspire_Male_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
file_name_f = ("Aspire_Female_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
control_m = pd.read_csv(file_name_m)
control_f = pd.read_csv(file_name_f)

fname = [0.7, 0.7, 1, 0.8, 0.8]
file_name_m = ("Aspire_Male_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
file_name_f = ("Aspire_Female_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
treatment_f = pd.read_csv(file_name_f)
treatment_m = pd.read_csv(file_name_m)

res = summary_cost(fname, control_m, control_f, treatment_m, treatment_f,'Pill Disutility')
summary_output.append(res[0])
appendix_output.append(res[1])
plot_output.append(res[2])

'''Analysis 10: Inclusion of Pill Disutility'''
time_horizon = 20
prog_cost = 0.13
discount_rate = 0.03

os.chdir('/Users/jarvis/Dropbox/Apps/HypertensionOutputs/PillDisFinal')
fname = [0.4, 0.3, 0, 0.8, 0.6]
file_name_m = ("Aspire_Male_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
file_name_f = ("Aspire_Female_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
control_m = pd.read_csv(file_name_m)
control_f = pd.read_csv(file_name_f)

fname = [0.7, 0.7, 0, 0.8, 0.8]
file_name_m = ("Aspire_Male_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
file_name_f = ("Aspire_Female_Cov_" + str(fname[0]) + "_Comp_" + str(fname[1]) + "_Pro_" + str(round(fname[2])) + "_Ini_" + str(fname[3]) + "_Per_" + str(
    fname[4]) + ".csv")
treatment_f = pd.read_csv(file_name_f)
treatment_m = pd.read_csv(file_name_m)

res = summary_cost(fname, control_m, control_f, treatment_m, treatment_f,'Pill Disutility with NPCDCS')
summary_output.append(res[0])
appendix_output.append(res[1])
plot_output.append(res[2])


os.chdir('/Users/jarvis/Dropbox/Apps/HypertensionOutputs/')

manuscript_results = pd.DataFrame(summary_output,
                                  columns=['Protocol', 'Coverage', 'Adherence', 'Initiation', 'Persistence', 'ICER', 'ICER_lower', 'ICER_upper', '% DALYs', '% DALYs_lower', '% DALYs_lower','HTN_lower', 'HTN_upper',
                                           'CVD_lower','CVD_upper', 'budget', 'annual htn','annual cvd', 'HTN percapita', 'CVD percapita','p_costsaving', 'p_negativeDALY'])

manuscript_results.to_csv('ConsolidatedSensitivityResult_30Aug.csv')

manuscript_text = pd.DataFrame(appendix_output, columns = ['Protocol', 'Coverage', 'Adherence', 'Initiation', 'Persistence','ICER', 'DALYs Averted', 'Increase in Cost','HTN Cost', 'CVD Cost', 'Budget 000s', 'Cost Saving Scenarios'])
manuscript_text.to_csv("ConsolidatedSensitivityText_30Aug.csv")

manuscript_plot = pd.DataFrame(plot_output, columns = ['Scenario', 'ICER', 'Cost', 'DALY', 'Error_Cost', 'Error_DALY'])
manuscript_plot.to_csv("ConsolidatedPlot_30Aug.csv")
