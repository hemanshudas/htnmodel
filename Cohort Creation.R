rm(list = ls())               #removing any variables in R's memory
options(max.print=10000)

# Processing Initial Data
processsage <- function(inp) {
  
  df <- z <- data.frame("id"=integer(),"age" = integer(),"sex"=integer(),"sbp" = double(),"dbp" = double(),"bmi" = double(),"smk" = integer(),
                         "iso"=character(),"year" = integer(),"htn_status"=character(),"facility"=integer(),stringsAsFactors=FALSE)
  
  for (row in 1 : nrow(inp)) {
    id <- inp[row,"id"]
    age <- inp[row,"q1011"]
    if (inp[row,"q1009"] == "Male") {     #Male - 0, Female - 1
      sex <- 0
    } else {
      sex <- 1
    }
    sbp <- (inp[row,"q2501_s"] +inp[row,"q2502_s"]+ inp[row,"q2503_s"])/3
    dbp <- (inp[row,"q2501_d"] +inp[row,"q2502_d"]+ inp[row,"q2503_d"])/3
    bmi <- inp[row,"q2507"] / ((0.01 * inp[row,"q2506"])^2)
    if (inp[row,"q3002"] == "Yes, daily" | inp[row,"q3002"] == "Yes, not daily"){
      smk <- 1
    } else {
      smk <- 0
    }
    if (sbp > 140) {
      if (inp[row,"q4060"] == "Yes") {
        htn_status = "Uncontrolled"
      } else {
        htn_status = "Unaware"
      }
    } else {
      if (inp[row,"q4060"] == "Yes") {
        htn_status = "Controlled"
      } else {
        htn_status = "Not hypertensive"
      }
    }
    if (inp[row,"q5004"] == "Public clinic" | inp[row,"q5004"] == "Public hospital") {
      facility = 0      # 0 for public sector
    } else {
      facility = 1      # 1 for any facility
    }
    iso <- "IND"
    year <- 2010
    
    z <- data.frame(id,sex,age,sbp,dbp,bmi,smk,iso,year,htn_status,facility)
    
    df <- rbind(df,z)
  }
  
  return(df)
}

#Generating age- and gender-specific cohort
cohortgen <- function(inp,sex,lower_age,samples,facility_choice=1){
  
  id <- lower_age*1000000
  
  htn <- non_htn <- data.frame("id"=integer(),"age" = integer(),"sex"=integer(),"sbp" = double(),"dbp" = double(),"bmi" = double(),"smk" = integer(),
                               "iso"=character(),"year" = integer(),"htn_status"=character(),"facility"=integer(),stringsAsFactors=FALSE)
  df <- z <- data.frame("id"=integer(),"age" = integer(),"sex"=integer(),"sbp" = double(),"bmi" = double(),"smk" = integer(),
                        "iso"=character(),"year" = integer(),stringsAsFactors=FALSE)
  count <- 0
  
  #Categorizing data into Hypertensive vs Not Hypertensive
  if (facility_choice == 0) {
    for (i in 1:nrow(inp)){
      if (inp[i,"sex"] == sex & inp[i,"age"] >= lower_age & inp[i,"age"] < (lower_age+5) & inp[i,"facility"] == 0 ) {
        count <- count + 1
        if (inp[i,"htn_status"]=="Not hypertensive"){
          non_htn <- rbind(non_htn,inp[i,])
        } else if (inp[i,"htn_status"]=="Unaware" | inp[i,"htn_status"]=="Uncontrolled" ){
          htn <- rbind(htn,inp[i,])
        }
      }
    }
  } else {
    for (i in 1:nrow(inp)){
      if (inp[i,"sex"] == sex & inp[i,"age"] >= lower_age & inp[i,"age"] < (lower_age+5)) {
        count <- count + 1
        if (inp[i,"htn_status"]=="Not hypertensive"){
          non_htn <- rbind(non_htn,inp[i,])
        } else if (inp[i,"htn_status"]=="Unaware" | inp[i,"htn_status"]=="Uncontrolled" ){
          htn <- rbind(htn,inp[i,])
        }
      }
    }
  }
  
  
  prob_htn <- 1- (nrow(non_htn)/count)    #Hypertension prevalence in the cohort
  
  for (i in 1:samples) {
    htn_decide <- sample(0:1, size=1, prob = c(prob_htn,1-prob_htn),replace=TRUE)           #Deciding if the person is hypertensive or not
    if (htn_decide == 0){
      line <- sample(1:nrow(htn),1,replace=TRUE)
      id <- id + 1
      age <- htn[line,"age"]
      sex <- htn[line,"sex"]
      sbp <- htn[line,"sbp"]
      bmi <- htn[line,"bmi"]
      smk <- htn[line,"smk"]
      iso <- "IND"
      year<- 2010
    } 
    else {
      line <- sample(1:nrow(non_htn),1,replace=TRUE)
      id <- id + 1
      age <- non_htn[line,"age"]
      sex <- non_htn[line,"sex"]
      sbp <- non_htn[line,"sbp"]
      bmi <- non_htn[line,"bmi"]
      smk <- non_htn[line,"smk"]
      iso <- "IND"
      year<- 2010
    }
    z <- data.frame(id,sex,age,sbp,bmi,smk,iso,year)
    df <- rbind(df,z)
  }
  return(df)
}

#Generating unified sex-based cohort
samplegen <- function(input,gender,n_samples,fac) {       #Gender: 0 fo Male, and 1 for Female, Facility: 0 for Public, 1 for All
  
  out <- s <- data.frame("id"=integer(),"age" = integer(),"sex"=integer(),"sbp" = double(),"bmi" = double(),"smk" = integer(),
                         "iso"=character(),"year" = integer(),stringsAsFactors=FALSE)
  
  for (i in 1:8)
  {
    age <- 40 + (i-1)*5
    if (gender==0) {
      s <- cohortgen(input,gender,age,round(n_samples*male_dist[i]),fac)  
    } else {
      s <- cohortgen(input,gender,age,round(n_samples*female_dist[i]),fac)
    }
    out <- rbind(out,s)
  }
  return(out)
}

# Franklin Implentation to increase blood pressure with age
franklin_sbpinc <- function (sample,age_till) {
  
  dfM1 <- sim_sample <- data.frame("id" = integer(),"age" = integer(),"sex" = character(),"sbp" = double(),
                                   "smk"= integer(), "bmi"= double(),"iso" = character(),"year"=integer(),stringsAsFactors=FALSE)
  
  sim_sample <- sample
  
  cur <- 0
  cur_calib <- 0
  slp <- 0
  slp_calib <- 0
  
  for( row in 1: nrow(sim_sample))
  {
    age <- sim_sample[row,"age"]
    sex <- sim_sample[row,"sex"]
    sbp <- sim_sample[row,"sbp"]
    smk <- sim_sample[row,"smk"]
    bmi <- sim_sample[row,"bmi"]
    id  <- sim_sample[row,"id"]
    iso <- "IND"
    year<- 2010
    
    if (age %% 5 != 0)
    {
      while (age %% 5 !=0)
      {
        dfM1 <- rbind(dfM1, data.frame(id,age,sex,sbp,smk,bmi,iso,year))
        if (sbp < 120) {
          cur <- 0.014
          cur_calib <- 0.35
          slp <- 0.57
          slp_calib <- 0.6
        } else if (sbp >= 120 & sbp < 140) {
          cur <- 0.007
          cur_calib <- 0.35
          slp <- 0.75
          slp_calib <- 0.6
        } else if (sbp >= 140 & sbp <160) {
          cur <- 0.001
          cur_calib <- 0.35
          slp <- 1.18
          slp_calib <- 0.45
        } else {
          cur <- 0.013
          cur_calib <- 0.35
          slp <- 1.97
          slp_calib <- 0.4
        }
        inc = 2*cur_calib*cur*(age-60) + slp_calib*slp
        sbp = sbp + inc
        age = age + 1
      }
    }
    while (age <= age_till)
    {
      dfM1 <- rbind(dfM1, data.frame(id,age,sex,sbp,smk,bmi,iso,year))
      if (sbp < 120) {
        cur <- 0.014
        cur_calib <- 0.35
        slp <- 0.57
        slp_calib <- 0.6
      } else if (sbp >= 120 & sbp < 140) {
        cur <- 0.007
        cur_calib <- 0.35
        slp <- 0.75
        slp_calib <- 0.6
      } else if (sbp >= 140 & sbp <160) {
        cur <- 0.001
        cur_calib <- 0.35
        slp <- 1.18
        slp_calib <- 0.45
      } else {
        cur <- 0.013
        cur_calib <- 0.35
        slp <- 1.97
        slp_calib <- 0.4
      }
      inc = (2*cur_calib*cur*(age-60) + slp_calib*slp)*5
      sbp = sbp + inc
      age <- age + 5
    }
  }
  return(dfM1)      
}

sim_inp <- read.csv("SAGE_Cleaned_19Oct.csv")   #Cleaned SAGE File as input

male_dist <- c(0.229,0.199,0.165,0.132,0.113,0.079,0.053,0.03)
female_dist <- c(0.224,0.199,0.158,0.136,0.114,0.081,0.055,0.033)

output <- processsage(sim_inp)

female_samp <- samplegen(output,1,100,1)
male_samp <- samplegen(output,0,100,1)

female_samples <- franklin_sbpinc(female_samp,age_till = 120)
male_samples <- franklin_sbpinc(male_samp,age_till = 120)

#Output files which will be input for thr Globorisk calculator
write.csv(female_samples,"Females_Globorisk_Input.csv")
write.csv(male_samples,"Males_Globorisk_Input.csv")


