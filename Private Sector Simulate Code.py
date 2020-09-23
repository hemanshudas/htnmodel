import os
import random
import time
from multiprocessing import Pool
import dropbox
import numpy
import pandas as pd

dbx = dropbox.Dropbox('')       #Enter the dropbox account id
dbx.users_get_current_account()

# Reading the input file - need to change based on gender
url_input_f = 'https://raw.githubusercontent.com/hemanshudas/htnmodel/master/Python%20Cohort%20Input/Female_PyInput.csv'
url_input_m = 'https://raw.githubusercontent.com/hemanshudas/htnmodel/master/Python%20Cohort%20Input/Male_PyInput.csv'

# CVD and Death Proportion

# For Males
url_cvdprop_m = 'https://raw.githubusercontent.com/hemanshudas/htnmodel/master/CVD%20Proportion/Male_MItoStroke.csv'
url_deaths_m = 'https://raw.githubusercontent.com/hemanshudas/htnmodel/master/Non%20CVD%20Deaths/Male_NonCVDDeaths.csv'

# For Females
url_cvdprop_f = 'https://raw.githubusercontent.com/hemanshudas/htnmodel/master/CVD%20Proportion/Female_MItoStroke.csv'
url_deaths_f = 'https://raw.githubusercontent.com/hemanshudas/htnmodel/master/Non%20CVD%20Deaths/Female_NonCVDDeaths.csv'

# Reading the treatment related risk files with '_f' referring to complete adherence, and '_p' referring to partial adherence
rr40_f = pd.read_csv(
    'https://raw.githubusercontent.com/hemanshudas/htnmodel/master/Relative%20Risk%20and%20Cost%20in%20Government%20Sector/RR_Age40-49_HighAdherence.csv')
rr50_f = pd.read_csv(
    'https://raw.githubusercontent.com/hemanshudas/htnmodel/master/Relative%20Risk%20and%20Cost%20in%20Government%20Sector/RR_Age50-59_HighAdherence.csv')
rr60_f = pd.read_csv(
    'https://raw.githubusercontent.com/hemanshudas/htnmodel/master/Relative%20Risk%20and%20Cost%20in%20Government%20Sector/RR_Age60-69_HighAdherence.csv')
rr70_f = pd.read_csv(
    'https://raw.githubusercontent.com/hemanshudas/htnmodel/master/Relative%20Risk%20and%20Cost%20in%20Government%20Sector/RR_Age70-79_HighAdherence.csv')
rr80_f = pd.read_csv(
    'https://raw.githubusercontent.com/hemanshudas/htnmodel/master/Relative%20Risk%20and%20Cost%20in%20Government%20Sector/RR_Age80-89_HighAdherence.csv')
rr40_p = pd.read_csv(
    'https://raw.githubusercontent.com/hemanshudas/htnmodel/master/Relative%20Risk%20and%20Cost%20in%20Government%20Sector/RR_Age40-49_MediumAdherence.csv')
rr50_p = pd.read_csv(
    'https://raw.githubusercontent.com/hemanshudas/htnmodel/master/Relative%20Risk%20and%20Cost%20in%20Government%20Sector/RR_Age50-59_MediumAdherence.csv')
rr60_p = pd.read_csv(
    'https://raw.githubusercontent.com/hemanshudas/htnmodel/master/Relative%20Risk%20and%20Cost%20in%20Government%20Sector/RR_Age60-69_MediumAdherence.csv')
rr70_p = pd.read_csv(
    'https://raw.githubusercontent.com/hemanshudas/htnmodel/master/Relative%20Risk%20and%20Cost%20in%20Government%20Sector/RR_Age70-79_MediumAdherence.csv')
rr80_p = pd.read_csv(
    'https://raw.githubusercontent.com/hemanshudas/htnmodel/master/Relative%20Risk%20and%20Cost%20in%20Government%20Sector/RR_Age80-89_MediumAdherence.csv')

Stroke_fatal_F = pd.read_csv(
    'https://raw.githubusercontent.com/hemanshudas/htnmodel/master/CVD_Case%20Fatality/ST_Female%20-%20Copy.csv')
MI_fatal_F = pd.read_csv(
    'https://raw.githubusercontent.com/hemanshudas/htnmodel/master/CVD_Case%20Fatality/MI_Female%20-%20Copy.csv')
Stroke_fatal_M = pd.read_csv(
    'https://raw.githubusercontent.com/hemanshudas/htnmodel/master/CVD_Case%20Fatality/ST_Male%20-%20Copy.csv')
MI_fatal_M = pd.read_csv(
    'https://raw.githubusercontent.com/hemanshudas/htnmodel/master/CVD_Case%20Fatality/MI_Male%20-%20Copy.csv')
start_time = time.time()


def complete_sim(target):
    def beta_gen(x_min, x_max, psa_seed):
        numpy.random.seed(psa_seed)
        mu = (x_max + x_min) / 2

        var = ((x_max - x_min) / 6) ** 2

        alpha = (((1 - mu) / var) - (1 / mu)) * (mu ** 2)
        beta = alpha * ((1 / mu) - 1)

        rv = numpy.random.beta(alpha, beta)

        return rv

    def beta_gen2(x_min, x_max, mu, psa_seed):
        numpy.random.seed(psa_seed)

        var = ((x_max - x_min) / 6) ** 2

        alpha = (((1 - mu) / var) - (1 / mu)) * (mu ** 2)
        beta = alpha * ((1 / mu) - 1)

        rv = numpy.random.beta(alpha, beta)

        return rv

    def normal_gen(mu, sd, psa_seed):
        numpy.random.seed(psa_seed)

        rv = numpy.random.normal(mu, sd)

        return rv

    def alloc(p_aware, p_start, p_persist, p_adhere, protocol_choice, set_seed):
        numpy.random.seed(set_seed)
        id = 0

        assigned_samples = []
        for i in range(stata_in.shape[0]):
            if pd.isna(stata_in.loc[i, "ten_y_cvd"]):
                stata_in.loc[i, "ten_y_cvd"] = stata_in.loc[i - 1, "ten_y_cvd"]

        for i in range(stata_in.shape[0]):
            if stata_in.loc[i, "id"] != id:
                hypertensive = status_aware = eligible_treat = on_treatment = treatment_start = 0
                id = stata_in.loc[i, "id"]
                age = stata_in.loc[i, "age"]
                sex = stata_in.loc[i, "sex"]
                sbp = stata_in.loc[i, "sbp"]
                smk = stata_in.loc[i, "smk"]
                bmi = stata_in.loc[i, "bmi"]
                ten_y_cvd = stata_in.loc[i, "ten_y_cvd"]

                if sbp >= 140:
                    hypertensive = 1
                    status_aware = int(numpy.random.choice([0, 1], size=1, replace=True, p=[1 - p_aware, p_aware]))
                else:
                    status_aware = 0

                if protocol_choice == 0:
                    if status_aware == 1:
                        if (sbp >= 160) | ((sbp >= 140) & (ten_y_cvd > 0.2)):
                            eligible_treat = 1
                    else:
                        eligible_treat = 0  # Seems unrequired
                else:
                    if status_aware == 1:
                        eligible_treat = 1
                    else:
                        eligible_treat = 0  # Seems unrequired

                if eligible_treat == 1:
                    treatment_start = int(numpy.random.choice([0, 1], size=1, replace=True, p=[1 - p_start, p_start]))
                else:
                    treatment_start = 0  # Seems unrequired

                if treatment_start == 1:
                    on_treatment = int(numpy.random.choice([0, 1, 2], size=1, replace=True,
                                                           p=[1 - p_persist, p_persist * (1 - p_adhere),
                                                              p_persist * p_adhere]))
                else:
                    on_treatment = 0  # Seems unrequired

            else:
                if ((hypertensive == 1) & (status_aware == 0)) | (
                        (hypertensive == 1) & (status_aware == 1) & (eligible_treat == 1)):
                    id = stata_in.loc[i - 1, "id"]
                    age = stata_in.loc[i, "age"]
                    sex = stata_in.loc[i, "sex"]
                    smk = stata_in.loc[i, "smk"]
                    sbp = stata_in.loc[i, "sbp"]
                    bmi = stata_in.loc[i, "bmi"]
                    ten_y_cvd = stata_in.loc[i, "ten_y_cvd"]
                elif (hypertensive == 1) & (status_aware == 1) & (eligible_treat == 0):
                    id = stata_in.loc[i, "id"]
                    age = stata_in.loc[i, "age"]
                    sex = stata_in.loc[i, "sex"]
                    smk = stata_in.loc[i, "smk"]
                    sbp = stata_in.loc[i, "sbp"]
                    bmi = stata_in.loc[i, "bmi"]
                    ten_y_cvd = stata_in.loc[i, "ten_y_cvd"]

                    if protocol_choice == 0:
                        if status_aware == 1:
                            if (sbp >= 160) | ((sbp >= 140) & (ten_y_cvd > 0.2)):
                                eligible_treat = 1
                        else:
                            eligible_treat = 0  # Seems unrequired
                    else:
                        if status_aware == 1:
                            eligible_treat = 1
                        else:
                            eligible_treat = 0  # Seems unrequired

                    if eligible_treat == 1:
                        treatment_start = int(
                            numpy.random.choice([0, 1], size=1, replace=True, p=[1 - p_start, p_start]))
                    else:
                        treatment_start = 0  # Seems unrequired

                    if treatment_start == 1:
                        on_treatment = int(numpy.random.choice([0, 1, 2], size=1, replace=True,
                                                               p=[1 - p_persist, p_persist * (1 - p_adhere),
                                                                  p_persist * p_adhere]))
                    else:
                        on_treatment = 0  # Seems unrequired
                else:
                    id = stata_in.loc[i, "id"]
                    age = stata_in.loc[i, "age"]
                    sex = stata_in.loc[i, "sex"]
                    smk = stata_in.loc[i, "smk"]
                    sbp = stata_in.loc[i, "sbp"]
                    bmi = stata_in.loc[i, "bmi"]
                    ten_y_cvd = stata_in.loc[i, "ten_y_cvd"]

                    if sbp >= 140:
                        hypertensive = 1
                        status_aware = int(numpy.random.choice([0, 1], size=1, replace=True, p=[1 - p_aware, p_aware]))
                    else:
                        status_aware = 0

                    if protocol_choice == 0:
                        if status_aware == 1:
                            if (sbp >= 160) | ((sbp >= 140) & (ten_y_cvd > 0.2)):
                                eligible_treat = 1
                        else:
                            eligible_treat = 0  # Seems unrequired
                    else:
                        if status_aware == 1:
                            eligible_treat = 1
                        else:
                            eligible_treat = 0  # Seems unrequired

                    if eligible_treat == 1:
                        treatment_start = int(
                            numpy.random.choice([0, 1], size=1, replace=True, p=[1 - p_start, p_start]))
                    else:
                        treatment_start = 0  # Seems unrequired

                    if treatment_start == 1:
                        on_treatment = int(numpy.random.choice([0, 1, 2], size=1, replace=True,
                                                               p=[1 - p_persist, p_persist * (1 - p_adhere),
                                                                  p_persist * p_adhere]))
                    else:
                        on_treatment = 0  # Seems unrequired

            assigned_samples.append(
                [id, age, sex, sbp, ten_y_cvd, smk, bmi, hypertensive, status_aware, eligible_treat, treatment_start,
                 on_treatment
                 ])

        output_samples = pd.DataFrame(assigned_samples,
                                      columns=["id", "age", "sex", "sbp", "ten_y_cvd", "smk", "bmi", "hypertensive",
                                               "status_aware",
                                               "eligible_treat", "treatment_start", "on_treatment"])

        return output_samples

    def count_treat():
        count_h = count_t = count_aw = count_el = count_a = 0
        id = 0

        for i in range(alloc_out.shape[0]):
            if alloc_out.loc[i, "id"] != id:

                if alloc_out.loc[i, "hypertensive"] == 1:
                    count_h = count_h + 1
                    if alloc_out.loc[i, "status_aware"] == 1:
                        count_aw = count_aw + 1
                        if alloc_out.loc[i, "eligible_treat"] == 1:
                            count_el = count_el + 1
                            if alloc_out.loc[i, "treatment_start"] == 1:
                                count_t = count_t + 1
                                if alloc_out.loc[i, "on_treatment"] == 2:
                                    count_a = count_a + 1
            id = alloc_out.loc[i, "id"]

        intervention_char = [count_t / count_h, count_a / count_t, count_h, count_aw, count_el, count_t, count_a]
        return intervention_char

    def age_prob(p_W_CVD, row):
        c_W = 0
        p_CVD_MI = beta_gen(cvdprop.loc[row, "Lower Value"], cvdprop.loc[row, "Upper Value"], alloc_iterate)
        p_W_MI = p_CVD_MI * p_W_CVD
        p_W_ST = (1 - p_CVD_MI) * p_W_CVD
        p_W_D = beta_gen(deaths.loc[row, "Lower Value"], deaths.loc[row, "Upper Value"], alloc_iterate)

        return p_W_MI, p_W_ST, p_W_D, c_W

    def riskred(sbp, p_W_CVD, p_CVD_MI, rr):
        if (sbp >= 140) & (sbp < 150):
            p_W_MI = p_W_CVD * p_CVD_MI * rr.loc[0, "rr.MI"]
            p_W_ST = p_W_CVD * (1 - p_CVD_MI) * rr.loc[0, "rr.ST"]
            c_W = cost_factor * rr.loc[0, "c.W"]
        elif (sbp >= 150) & (sbp < 160):
            p_W_MI = p_W_CVD * p_CVD_MI * rr.loc[1, "rr.MI"]
            p_W_ST = p_W_CVD * (1 - p_CVD_MI) * rr.loc[1, "rr.ST"]
            c_W = cost_factor * rr.loc[1, "c.W"]
        elif (sbp >= 160) & (sbp < 170):
            p_W_MI = p_W_CVD * p_CVD_MI * rr.loc[2, "rr.MI"]
            p_W_ST = p_W_CVD * (1 - p_CVD_MI) * rr.loc[2, "rr.ST"]
            c_W = cost_factor * rr.loc[2, "c.W"]
        elif (sbp >= 170) & (sbp < 180):
            p_W_MI = p_W_CVD * p_CVD_MI * rr.loc[3, "rr.MI"]
            p_W_ST = p_W_CVD * (1 - p_CVD_MI) * rr.loc[3, "rr.ST"]
            c_W = cost_factor * rr.loc[3, "c.W"]
        elif sbp >= 180:
            p_W_MI = p_W_CVD * p_CVD_MI * rr.loc[4, "rr.MI"]
            p_W_ST = p_W_CVD * (1 - p_CVD_MI) * rr.loc[4, "rr.ST"]
            c_W = cost_factor * rr.loc[4, "c.W"]

        return p_W_MI, p_W_ST, c_W

    def age_prob_rr(sbp, p_W_CVD, rr, row, on_treatment):
        p_CVD_MI = beta_gen(cvdprop.loc[row, "Lower Value"], cvdprop.loc[row, "Upper Value"], alloc_iterate)
        p_W_D = beta_gen(deaths.loc[row, "Lower Value"], deaths.loc[row, "Upper Value"], alloc_iterate)
        rr_MI_ST = riskred(sbp, p_W_CVD, p_CVD_MI, rr)
        p_W_MI = rr_MI_ST[0]
        p_W_ST = rr_MI_ST[1]
        if on_treatment == 2:
            c_W = rr_MI_ST[2]
        else:
            c_W = rr_MI_ST[2] * beta_gen2(0.5, 0.8, 0.7, alloc_iterate)

        return p_W_MI, p_W_ST, p_W_D, c_W

    def colAdd(age, sbp, ten_y_cvd, on_treatment):
        p_W_CVD_yr = - (numpy.log(1 - ten_y_cvd)) / 10
        p_W_CVD = (1 - numpy.exp(-p_W_CVD_yr / 12)) * (1 - risk_reduction)      #Conversion to monthly risk using Sonnenberg's method

        if on_treatment == 0:
            if age != 90:
                if (age >= 40) & (age <= 44):
                    tr_prob = age_prob(p_W_CVD, 0)
                elif (age >= 45) & (age <= 49):
                    tr_prob = age_prob(p_W_CVD, 1)
                elif (age >= 50) & (age <= 54):
                    tr_prob = age_prob(p_W_CVD, 2)
                elif (age >= 55) & (age <= 59):
                    tr_prob = age_prob(p_W_CVD, 3)
                elif (age >= 60) & (age <= 64):
                    tr_prob = age_prob(p_W_CVD, 4)
                elif (age >= 65) & (age <= 69):
                    tr_prob = age_prob(p_W_CVD, 5)
                elif (age >= 70) & (age <= 74):
                    tr_prob = age_prob(p_W_CVD, 6)
                elif (age >= 75) & (age <= 79):
                    tr_prob = age_prob(p_W_CVD, 7)
                elif (age >= 80) & (age <= 84):
                    tr_prob = age_prob(p_W_CVD, 8)
                else:
                    tr_prob = age_prob(p_W_CVD, 9)
        else:
            if on_treatment == 2:
                rr40 = rr40_f
                rr50 = rr50_f
                rr60 = rr60_f
                rr70 = rr70_f
                rr80 = rr80_f
            else:
                rr40 = rr40_p
                rr50 = rr50_p
                rr60 = rr60_p
                rr70 = rr70_p
                rr80 = rr80_p
            if age != 90:
                if (age >= 40) & (age <= 44):
                    tr_prob = age_prob_rr(sbp, p_W_CVD, rr40, 0, on_treatment)
                elif (age >= 45) & (age <= 49):
                    tr_prob = age_prob_rr(sbp, p_W_CVD, rr40, 1, on_treatment)
                elif (age >= 50) & (age <= 54):
                    tr_prob = age_prob_rr(sbp, p_W_CVD, rr50, 2, on_treatment)
                elif (age >= 55) & (age <= 59):
                    tr_prob = age_prob_rr(sbp, p_W_CVD, rr50, 3, on_treatment)
                elif (age >= 60) & (age <= 64):
                    tr_prob = age_prob_rr(sbp, p_W_CVD, rr60, 4, on_treatment)
                elif (age >= 65) & (age <= 69):
                    tr_prob = age_prob_rr(sbp, p_W_CVD, rr60, 5, on_treatment)
                elif (age >= 70) & (age <= 74):
                    tr_prob = age_prob_rr(sbp, p_W_CVD, rr70, 6, on_treatment)
                elif (age >= 75) & (age <= 79):
                    tr_prob = age_prob_rr(sbp, p_W_CVD, rr60, 7, on_treatment)
                elif (age >= 80) & (age <= 84):
                    tr_prob = age_prob_rr(sbp, p_W_CVD, rr80, 8, on_treatment)
                else:
                    tr_prob = age_prob_rr(sbp, p_W_CVD, rr80, 9, on_treatment)

        return tr_prob[0], tr_prob[1], tr_prob[2], tr_prob[3]

    def exp_cohort():
        rr_samples = []
        for i in range(alloc_out.shape[0]):
            char_inp = []
            char_inp = alloc_out.iloc[i, 0:5]
            on_treatment = alloc_out.loc[i, "on_treatment"]

            if char_inp[1] != 90:
                rr_values = colAdd(char_inp[1], char_inp[3], char_inp[4], on_treatment)
                rr_samples.append(
                    [char_inp[0], char_inp[1], char_inp[2], char_inp[3], on_treatment, rr_values[0], rr_values[1],
                     rr_values[2], rr_values[3]])

        markos_samples = pd.DataFrame(rr_samples,
                                      columns=["id", "age", "sex", "sbp", "on_treatment", "p_W_MI", "p_W_ST", "p_W_D",
                                               "c_W"])

        return markos_samples

    def simulate_cohort(time_horizon, mi_fatal, stroke_fatal, int_composite, seed_x):
        # Start of function
        count_people = 0.00
        row = 0
        id_ind = 0

        # Initializing cohort result variables
        costs = utility = 0
        tot_D = tot_cD = 0
        trt_count = 0
        count_CVD = 0
        costofHTN = costofCVD = costofChronic = 0

        model_run_time = time_horizon

        prop_treat_access = 0.835
        postST = stroke_fatal  # Fatality due to stroke
        postMI = mi_fatal  # Fatality due to MI
        if gender == 0:         #Fatality due to CVD among CVD patients - Bronnen Hansen
            prop_cD_MI = 0.76
            prop_cD_ST = 0.634
        else:
            prop_cD_MI = 0.69
            prop_cD_ST = 0.723

        # Cost of each state in US $
        c_W = c_D = c_cD = 0
        c_MI = 952 * (1.03 ** 3)
        c_ST = 860 * (1.03 ** 3)
        c_postMI = (3.41 * (1.03 ** 3)) + numpy.random.choice([18.917, 37.834], 1, replace = True, p= [0.5, 0.5])[0]
        c_postST = (5.08 * (1.03 ** 3)) + numpy.random.choice([18.917, 37.834], 1, replace = True, p= [0.5, 0.5])[0]

        # AntiHTN treatment initiation
        #c_consult = normal_gen(2.22 *(1.03 ** 4), 0.24*(1.03 ** 4), alloc_iterate)  # Unit cost of outpatient consultation at PHCs and CHCs
        c_consult = normal_gen(1.95, 0.27, alloc_iterate)
        c_HTNTests = 2.27  # Cost of tests based on CHGS prices
        treat_ini_consults = 6  # Number of consultations in the first 6 months of treatment
        yearly_consults = 4  # Number of consultations in a year

        # Disability weight of each state
        u_W = 0
        u_MI = beta_gen2(0.288, 0.579, 0.432, alloc_iterate)
        u_ST = beta_gen2(0.377, 0.707, 0.57, alloc_iterate)
        u_postMI = beta_gen2(0.02, 0.24, 0.08, alloc_iterate)
        u_postST = beta_gen2(0.01, 0.437, 0.135, alloc_iterate)
        u_MIST = beta_gen2(0.985, 0.992, 0.989, alloc_iterate)
        u_postMIST = beta_gen2(0.11, 0.437, 0.242, alloc_iterate)
        u_D = u_cD = 1
        check_ids = []

        # Recurrence of acute CVD events while in acute phase
        r_MI_ST2 = beta_gen2(0.0045, 0.0075, 0.006, alloc_iterate)
        r_MI_MI2 = beta_gen2(0.0099, 0.0141, 0.012, alloc_iterate)
        r_ST_ST2 = beta_gen(0.1, 0.2, alloc_iterate)

        p_MI_MI2 = 1 - numpy.exp(-r_MI_MI2 / 12)
        p_MI_ST2 = 1 - numpy.exp(-r_MI_ST2 / 12)
        p_ST_ST2 = 1 - numpy.exp(-r_ST_ST2 / 12)
        p_ST_MI2 = 0

        # Recurrence of acute CVD events while in chronic phase
        r_postMI_MI2 = beta_gen(0.073, 0.085, alloc_iterate)
        r_postMI_ST2 = beta_gen(0.012, 0.016, alloc_iterate)
        r_postST_MI2 = beta_gen(0.038, 0.048, alloc_iterate)
        r_postST_ST2 = beta_gen(0.033, 0.041, alloc_iterate)

        p_postMI_MI2 = 1 - numpy.exp(-r_postMI_MI2 / 12)
        p_postMI_ST2 = 1 - numpy.exp(-r_postMI_ST2 / 12)
        p_postST_MI2 = 1 - numpy.exp(-r_postST_MI2 / 12)
        p_postST_ST2 = 1 - numpy.exp(-r_postST_ST2 / 12)

        def probability(m_s, ind_loc):
            v_p_it = numpy.array([[numpy.nan] * (n_s + 1) for i in range(n_i)])

            p_W_MI = markos.loc[ind_loc, "p_W_MI"]
            p_W_ST = markos.loc[ind_loc, "p_W_ST"]

            p_W_D = markos.loc[ind_loc, "p_W_D"]
            rr_recurrence = 1.5  # Increased risk of fatality

            # Row counter for understanding where to fetch fatality data from
            ind_age = markos.loc[ind_loc, "age"]

            if (ind_age >= 40) & (ind_age < 45):
                row_fetch = 0
            elif (ind_age >= 45) & (ind_age < 50):
                row_fetch = 1
            elif (ind_age >= 50) & (ind_age < 55):
                row_fetch = 2
            elif (ind_age >= 55) & (ind_age < 59):
                row_fetch = 3
            elif (ind_age >= 60) & (ind_age < 65):
                row_fetch = 4
            elif (ind_age >= 65) & (ind_age < 70):
                row_fetch = 5
            elif (ind_age >= 70) & (ind_age < 75):
                row_fetch = 6
            elif (ind_age >= 75) & (ind_age < 80):
                row_fetch = 7
            else:
                row_fetch = 8

            # Calculating the probability of death from MI & Stroke for each individual
            p_MI_cD = postMI.loc[row_fetch, "C1"]
            p_postMI_aD = postMI.loc[row_fetch, "C2"]
            p_postMI_cD = prop_cD_MI * p_postMI_aD
            p_postMI_D = (1 - prop_cD_MI) * p_postMI_aD

            p_ST_cD = postST.loc[row_fetch, "C1"]
            p_postST_aD = postST.loc[row_fetch, "C2"]
            p_postST_cD = prop_cD_ST * p_postST_aD
            p_postST_D = (1 - prop_cD_ST) * p_postST_aD

            # Calculating the probability of death from recurring MI & Stroke for each individual
            r_postMI2_aD2 = -numpy.log(1 - p_postMI_aD) * rr_recurrence
            r_postST2_aD2 = -numpy.log(1 - p_postST_aD) * rr_recurrence
            p_postMI2_aD = 1 - numpy.exp(-r_postMI2_aD2)
            p_postST2_aD = 1 - numpy.exp(-r_postST2_aD2)

            p_postMI2_cD2 = prop_cD_MI * p_postMI2_aD
            p_postMI2_D = (1 - prop_cD_MI) * p_postMI2_aD
            p_postST2_cD2 = prop_cD_ST * p_postST2_aD
            p_postST2_D = (1 - prop_cD_ST) * p_postST2_aD

            p_MI_postMI = 1 - (p_MI_cD + p_MI_MI2 + p_MI_ST2)
            p_ST_postST = 1 - (p_ST_cD + p_ST_ST2 + p_ST_MI2)

            p_postMI_postMI = 1 - (p_postMI_MI2 + p_postMI_ST2 + p_postMI_aD)
            p_postST_postST = 1 - (p_postST_ST2 + p_postST_MI2 + p_postST_aD)

            p_MI2_postMI2 = 1 - (p_MI_MI2 + p_MI_ST2 + p_MI_cD)
            p_ST2_postST2 = 1 - (p_ST_ST2 + p_ST_MI2 + p_ST_cD)

            p_postMI2_postMI2 = 1 - (p_postMI_MI2 + p_postMI_ST2 + p_postMI2_aD)
            p_postST2_postST2 = 1 - (p_postST_MI2 + p_postST_ST2 + p_postST2_aD)

            p_W_W = 1 - (p_W_MI + p_W_ST + p_W_D)
            p_W_MI_access = prop_treat_access * p_W_MI
            p_W_MI_noaccess = (1 - prop_treat_access) * p_W_MI

            if m_s == 'W':
                v_p_it = [p_W_W, p_W_MI_access, p_W_ST, 0, 0, 0, 0, 0, 0, p_W_MI_noaccess, p_W_D]
            elif m_s == 'MI':
                v_p_it = [0, 0, 0, p_MI_postMI, 0, p_MI_MI2, p_MI_ST2, 0, 0, p_MI_cD, 0]
            elif m_s == 'ST':
                v_p_it = [0, 0, 0, 0, p_ST_postST, p_ST_MI2, p_ST_ST2, 0, 0, p_ST_cD, 0]
            elif m_s == "postMI":
                v_p_it = [0, 0, 0, p_postMI_postMI, 0, p_postMI_MI2, p_postMI_ST2, 0, 0, p_postMI_cD, p_postMI_D]
            elif m_s == "postST":
                v_p_it = [0, 0, 0, 0, p_postST_postST, p_postST_MI2, p_postST_ST2, 0, 0, p_postST_cD, p_postST_D]
            elif m_s == "MI2":
                v_p_it = [0, 0, 0, 0, 0, p_MI_MI2, p_MI_ST2, p_MI2_postMI2, 0, p_MI_cD, 0]
            elif m_s == "ST2":
                v_p_it = [0, 0, 0, 0, 0, p_ST_MI2, p_ST_ST2, 0, p_ST2_postST2, p_ST_cD, 0]
            elif m_s == "postMI2":
                v_p_it = [0, 0, 0, 0, 0, p_postMI_MI2, p_postMI_ST2, p_postMI2_postMI2, 0, p_postMI2_cD2, p_postMI2_D]
            elif m_s == "postST2":
                v_p_it = [0, 0, 0, 0, 0, p_postST_MI2, p_postST_ST2, 0, p_postST2_postST2, p_postST2_cD2, p_postST2_D]
            elif m_s == "cD":
                v_p_it = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
            else:
                v_p_it = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

            return v_p_it

        def micro_sim(csv_counter, seed):
            v_dwc = 1 / (1 + d_c) ** numpy.array(range(0, n_t + 1))
            v_dwu = 1 / (1 + d_u) ** numpy.array(range(0, n_t + 1))

            m_M = numpy.array([[numpy.nan] * (n_t + 1) for i in range(n_i)])
            cost_HTN = cost_CVD = cost_Chronic = m_C = m_U = numpy.array([[0] * (n_t + 1) for i in range(n_i)],
                                                                         dtype=float)

            m_M = m_M.astype(str)
            m_C = m_C.astype(float)
            m_U = m_U.astype(float)
            cost_HTN = cost_HTN.astype(float)
            cost_CVD = cost_CVD.astype(float)
            cost_Chronic = cost_Chronic.astype(float)

            m_M[0, 0] = v_M_1

            flag_CVD_event = 0

            for i in range(n_i):
                z = csv_counter  # CSV Store updates row number back to the initial row number for other profiles with same ID
                ctr2 = ctr       # ctr2 updates ctr back to initial value for other profiles with same ID
                MI_months = 0    # Number of months in chronic IHD state
                ST_months = 0    # Number of months in chronic stroke states

                numpy.random.seed(int(seed))

                c_addon = 0
                if markos.loc[csv_counter, "on_treatment"] != 0:
                    c_addon = c_consult * (treat_ini_consults - yearly_consults / 2) + c_HTNTests
                m_C[i, 0] = c_addon  # Check if correct
                m_U[i, 0] = 0
                cost_HTN[i, 0] = m_C[i, 0]
                cost_Chronic[i, 0] = cost_CVD[i, 0] = 0

                for t in range(n_t):  # Different than the R code
                    if ctr2 != 0:
                        if ((t % 12) == 0) & (t != 0):
                            z = z + 1
                            ctr2 = ctr2 - 1
                        else:
                            z = z
                    else:
                        if (((t - months_till_5) % 60) == 0) & (t != 0):
                            z = z + 1
                        else:
                            z = z

                    v_p = probability(m_M[i, t], z)                                     # Transition probabilities at cycle t
                    temp_state = numpy.random.choice(v_n, size=1, replace=True, p=v_p)  # Storing the array output
                    m_M[i, t + 1] = temp_state[0]

                    if (m_M[i, t + 1] == "MI") | (m_M[i, t + 1] == "ST") | (
                            (m_M[i, t] == "W") & (m_M[i, t + 1] == "cD")):
                        flag_CVD_event = 1

                    # Determining the cost for each state
                    m_cost = m_M[i, t]
                    c_cW = 0
                    c_drugs = markos.loc[z, "c_W"]
                    if m_cost == 'W':
                        if markos.loc[z, "on_treatment"] != 0:
                            c_cW = c_drugs + (c_consult * yearly_consults / 12)
                        else:
                            c_cW = 0

                    options_c = {'W': c_cW, 'MI': c_MI, 'ST': c_ST, 'postMI': c_postMI, 'postST': c_postST, 'MI2': c_MI,
                                 'ST2': c_ST, 'postMI2': c_postMI,
                                 'postST2': c_postST, 'cD': c_cD, 'D': c_D}

                    c_it = options_c[m_cost]
                    m_C[i, t + 1] = c_it

                    # Determining the disability for the cycle
                    options_u = {'W': u_W, 'MI': u_MI, 'ST': u_ST, 'postMI': u_postMI, 'postST': u_postST,
                                 'MI2': u_MIST, 'ST2': u_MIST, 'postMI2': u_postMIST,
                                 'postST2': u_postMIST, 'cD': u_cD, 'D': u_D}

                    m_util = m_M[i, t]
                    u_it = options_u[m_util]
                    m_U[i, t + 1] = u_it

                    # Desegregating costs based on type of health state
                    if m_M[i, t + 1] == "W":
                        cost_HTN[i, t + 1] = m_C[i, t + 1]
                        cost_Chronic[i, t + 1] = cost_CVD[i, t + 1] = 0
                    elif (m_M[i, t + 1] == "MI") | (m_M[i, t + 1] == "ST") | (m_M[i, t + 1] == "MI2") | (
                            m_M[i, t + 1] == "ST2"):
                        cost_CVD[i, t + 1] = m_C[i, t + 1]
                        cost_Chronic[i, t + 1] = cost_HTN[i, t + 1] = 0
                    else:
                        cost_Chronic[i, t + 1] = m_C[i, t + 1]
                        cost_HTN[i, t + 1] = cost_CVD[i, t + 1] = 0

            # Calculating the number of CVD related deaths and other cause deaths
            cd_flag = d_flag = 0
            if m_M[0, n_t] == "cD":
                cd_flag = 1
            if m_M[0, n_t] == "D":
                d_flag = 1

            tc = numpy.sum(numpy.dot(m_C, v_dwc))
            tu = numpy.sum(numpy.dot(m_U, v_dwu))
            tcost_HTN = numpy.sum(numpy.dot(cost_HTN, v_dwc))
            tcost_CVD = numpy.sum(numpy.dot(cost_CVD, v_dwu))
            tcost_Chronic = numpy.sum(numpy.dot(cost_Chronic, v_dwc))

            return tc, tu, z, cd_flag, d_flag, flag_CVD_event, tcost_HTN, tcost_CVD, tcost_Chronic

        while row < markos.shape[0]:

            id_ind = markos.loc[row, "id"]
            csv_counter = row  # Row number counter for entire csv
            seed = seed_x + id_ind

            tc_hat_total = tu_hat_total = 0  # Initiating cost and DALY counters for each person
            HTN_cost = CVD_cost = Chronic_cost = 0

            count_people = count_people + 1

            ini_age = markos.loc[csv_counter, "age"]
            end_age = ini_age + model_run_time  # Determining when the model stops for each person

            if (markos.loc[
                    csv_counter, "age"] % 10) > 5:  # Ctr is the number of years to reach a age which is multiple of 5
                ctr = 10 - (markos.loc[csv_counter, "age"] % 10)
            else:
                if (markos.loc[csv_counter, "age"] % 10) == 0:
                    ctr = 0
                else:
                    ctr = 5 - (markos.loc[csv_counter, "age"] % 10)

            months_till_5 = ctr * 12

            n_i = 1  # Number of individuals - Legacy code - remains 1 here.
            v_n = ["W", "MI", "ST", "postMI", "postST", "MI2", "ST2", "postMI2", "postST2", "cD",
                   "D"]  # Markov states in the model
            n_s = len(v_n)
            v_M_1 = "W"  # Every individual starts at Well state at the start of model
            d_c = d_u = discount_rate / 12  # Equal discounting of cost and DALYs by 3% p.a

            n_t = model_run_time * 12  # Number of cycles in the model # Different from R file

            markos_res = micro_sim(csv_counter, seed)

            tc_hat_total = tc_hat_total + markos_res[0]
            tu_hat_total = tu_hat_total + markos_res[1]
            HTN_cost = HTN_cost + markos_res[6]
            CVD_cost = CVD_cost + markos_res[7]
            Chronic_cost = Chronic_cost + markos_res[8]

            csv_counter = markos_res[2]

            while markos.loc[csv_counter - 1, "id"] == markos.loc[csv_counter, "id"]:
                csv_counter = csv_counter + 1
                if csv_counter >= markos.shape[0]:
                    break

            tot_cD = tot_cD + markos_res[3]
            tot_D = tot_D + markos_res[4]

            row = csv_counter

            costs = costs + tc_hat_total
            utility = utility + tu_hat_total
            count_CVD = count_CVD + markos_res[5]

            costofHTN = costofHTN + HTN_cost
            costofCVD = costofCVD + CVD_cost
            costofChronic = costofChronic + Chronic_cost

        avgcost = costs / count_people
        avgdalys = utility / (12 * count_people)
        tot_aD = tot_cD + tot_D
        avgHTNcost = costofHTN / count_people
        avgCVDcost = costofCVD / count_people
        avgChroniccost = costofChronic / count_people

        coverage = int_composite[0]
        compliance = int_composite[1]

        return coverage, compliance, avgcost, avgdalys, count_CVD, tot_cD, tot_aD, avgHTNcost, avgCVDcost, avgChroniccost

    # Inputs for simulation
    sim_time_horizon = 20
    gender = target[5]  # 1 for Female and 0 for Male
    protocol = target[0]
    treatment_initiation = target[3]
    treatment_persistence = target[4]

    # Calculated parameters
    status_awareness = target[1] / treatment_initiation
    medication_adherence = target[2] / treatment_persistence

    start_trials = target[6]
    n_trials = target[7]

    if gender == 1:
        stata_in = pd.read_csv(url_input_f)
        cvdprop = pd.read_csv(url_cvdprop_f)
        deaths = pd.read_csv(url_deaths_f)
        gender_name = 'Female'
    else:
        stata_in = pd.read_csv(url_input_m)
        cvdprop = pd.read_csv(url_cvdprop_m)
        deaths = pd.read_csv(url_deaths_m)
        gender_name = 'Male'

    sim_output = []
    for alloc_iterate in range(start_trials, start_trials + n_trials):
        alloc_out = alloc(status_awareness, treatment_initiation, treatment_persistence, medication_adherence, protocol,
                          set_seed=alloc_iterate)
        composite = count_treat()
        risk_reduction = 0
        cost_factor = 1
        discount_rate = 0.03
        print(alloc_iterate)
        markos = exp_cohort()
        sample_output = simulate_cohort(sim_time_horizon, MI_fatal_M, Stroke_fatal_M, composite, seed_x=alloc_iterate)
        sim_output.append(
            [sample_output[0], sample_output[1], sample_output[2], sample_output[3], sample_output[4], sample_output[5],
             sample_output[6],
             sample_output[7], sample_output[8], sample_output[9]])

    final_output = pd.DataFrame(sim_output,
                                columns=['Coverage', 'Compliance', 'Average Cost', 'Average DALYs', 'CVD Count',
                                         'CVD Deaths', 'Total Deaths',
                                         'Average HTN Cost',
                                         'Average CVD Cost', 'Average Chronic Cost'])

    file_name = ("/28Aug_AWS3_Pvt_Final/Aspire_" + str(target[6]) + "_" + gender_name + "_Cov_" + str(target[1]) + "_Comp_" + str(
        target[2]) + "_Pro_" + str(protocol) + "_Ini_" + str(treatment_initiation) + "_Per_" + str(
        treatment_persistence) + ".csv")

    dbx.files_upload(final_output.to_csv().encode(), file_name)


def run_in_parallel():
    ranges = [[0, 0.4, 0.3, 0.8, 0.6, 1],
              [1, 0.7, 0.7, 0.8, 0.8, 1],
              [0, 0.7, 0.7, 0.8, 0.8, 1],
              [0, 0.4, 0.3, 0.8, 0.6, 0],
              [1, 0.7, 0.7, 0.8, 0.8, 0],
              [0, 0.7, 0.7, 0.8, 0.8, 0]]

    pool = Pool(processes=len(ranges))
    pool.map(complete_sim, ranges)


if __name__ == '__main__':
    run_in_parallel()

