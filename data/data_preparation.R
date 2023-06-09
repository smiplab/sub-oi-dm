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
  filter(stim == "pict")

df_chen$sub[df_chen$sub == 29][1:112] <- 1234
df_chen$sub[df_chen$sub == 34][1:112] <- 4321

sum <- df_chen %>% group_by(sub) %>% summarise(N = n())

df_chen %<>% 
  rename(id = sub,
         new_item = item,
         new_resp = response,
         rt = RT) %>% 
  select(id, new_item, new_resp, correct, rt) %>% 
  mutate(new_item = ifelse(new_item == "new", 1, 0),
         new_resp = ifelse(new_resp == "new", 1, 0)) %>% 
  group_by(id) %>%
  mutate(trial_filter = is_outlier(rt)) %>% 
  ungroup() %>% 
  mutate(rt = ifelse(trial_filter, NA, rt)) %>%
  select(-trial_filter) %>% 
  filter(!(id %in% c(51, 52, 101))) %>% 
  mutate(id = dense_rank(id)) %>% 
  group_by(id) %>% 
  mutate(trial = 1:112) %>% 
  arrange(id) %>% 
  write_csv("../application/yes_no_task/data/yes_no_data_picture.csv")

sum <- df_chen %>% group_by(id) %>% summarise(N = n())

#---------------------------------------------------------------#
# YES NO TASK
#---------------------------------------------------------------#
df_chen <- read_csv("chen-2019-4a.csv") %>% 
  filter(stim == "word")

df_chen$sub[df_chen$sub == 29][1:112] <- 1234
df_chen$sub[df_chen$sub == 34][1:112] <- 4321

sum <- df_chen %>% group_by(sub) %>% summarise(N = n())

df_chen %<>% 
  rename(id = sub,
         new_item = item,
         new_resp = response,
         rt = RT) %>% 
  select(id, new_item, new_resp, correct, rt) %>% 
  mutate(new_item = ifelse(new_item == "new", 1, 0),
         new_resp = ifelse(new_resp == "new", 1, 0)) %>% 
  group_by(id) %>%
  mutate(trial_filter = is_outlier(rt)) %>% 
  ungroup() %>% 
  mutate(rt = ifelse(trial_filter, NA, rt)) %>%
  select(-trial_filter) %>% 
  filter(!(id %in% c(51, 52, 101))) %>% 
  mutate(id = dense_rank(id)) %>% 
  group_by(id) %>% 
  mutate(trial = 1:112) %>% 
  arrange(id) %>% 
  write_csv("../application/yes_no_task/data/yes_no_data_words.csv")

sumsum <- df_chen %>% 
  group_by(id) %>% 
  summarise(mean_rt = mean(rt, na.rm = T))

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
  write_csv("../application/2afc_task/data/2afc_data.csv")

summary <- df_schnuerch %>% 
  group_by(id) %>% 
  summarise(n = n(),
            rt_mean = mean(rt, na.rm = T))
