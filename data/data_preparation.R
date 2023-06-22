library(tidyverse)
library(magrittr)


setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

df_chen <- read_csv("chen-2019-4a.csv")

df_chen <- df_chen %>% 
  filter(stim == "word")

#---------------------------------------------------------------

is_outlier <- function(x) {
  lower <- boxplot(log(x))$stats[1]
  upper <- boxplot(log(x))$stats[5]
  return(!between(log(x), lower, upper))
}

df_schnuerch <- read_csv("schnuerch-BA_oi_data.csv") %>% 
  filter(condition == "hehe") %>% 
  select(code, time, correct) %>% 
  rename(id = code,
         rt = time) %>% 
  mutate(rt = rt / 1000,
         id = dense_rank(id)) %>% 
  group_by(id) %>% 
  mutate(trial_filter = is_outlier(rt)) %>% 
  ungroup() %>% 
  mutate(rt = ifelse(correct == 1, rt, -rt)) %>% 
  filter(!trial_filter) %>% 
  select(-correct,
         -trial_filter)






  