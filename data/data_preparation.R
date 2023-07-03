library(tidyverse)
library(magrittr)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

is_outlier <- function(x) {
  lower <- boxplot(log(x))$stats[1]
  upper <- boxplot(log(x))$stats[5]
  return(!between(log(x), lower, upper))
}

#---------------------------------------------------------------#
# YES NO TASK
#---------------------------------------------------------------#
df_chen <- read_csv("chen-2019-4a.csv") %>% 
  filter(stim == "word") %>% 
  rename(id = sub,
         new_item = item,
         new_resp = response,
         rt = RT) %>% 
  select(id, new_item, new_resp, correct, rt) %>% 
  mutate(new_item = ifelse(new_item == "new", 1, 0),
         new_resp = ifelse(new_resp == "new", 1, 0),
         id = dense_rank(id)) %>% 
  group_by(id) %>%
  mutate(trial_filter = is_outlier(rt)) %>% 
  ungroup() %>% 
  mutate(rt = ifelse(trial_filter, NA, rt)) %>%
  select(-trial_filter) %>% 
  write_csv("../application/2afc_task/data/yes_no_data.csv")

#---------------------------------------------------------------#
# 2AFC
#---------------------------------------------------------------#
df_schnuerch <- read_csv("schnuerch-BA_oi_data.csv") %>% 
  filter(condition == "hehe",
         type == "hetero") %>% 
  select(code, oldPic, answer, correct, time) %>% 
  rename(id = code,
         correct_resp = oldPic,
         resp = answer,
         rt = time) %>% 
  mutate(rt = rt / 1000,
         id = dense_rank(id)) %>% 
  group_by(id) %>%
  mutate(trial_filter = is_outlier(rt)) %>%
  ungroup() %>%
  mutate(rt = ifelse(trial_filter, NA, rt)) %>%
  select(-trial_filter) %>%
  write_csv("../application/yes_no_task//data/2afc.csv")

summary <- df_schnuerch %>% 
  drop_na(rt) %>% 
  group_by(id) %>% 
  summarise(n = n())
