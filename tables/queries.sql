-- Get number of entries in each age group and sex that had a diagnosis and were detected as either benign or malignant
-- Ensure there is enough data for all 10 models
-- 
-- Currently producing the following:
-- 
-- Age_0_10_male  Age_0_10_female  Age_10_25_male  Age_10_25_female  Age_25_40_male  Age_25_40_female  Age_40_60_male  Age_40_60_female  Age_60_plus_male  Age_60_plus_female
-- -------------  ---------------  --------------  ----------------  --------------  ----------------  --------------  ----------------  ----------------  ------------------
-- 1143           686              5855            3876              3476            4409              7235            7098              6526              3729
-- 
-- TOTAL = 44033
-- 

SELECT
    SUM(CASE WHEN age_approx >= 0 AND age_approx <= 10 AND sex = "male" THEN 1 ELSE 0 END) AS Age_0_10_male,
    SUM(CASE WHEN age_approx >= 0 AND age_approx <= 10 AND sex = "female" THEN 1 ELSE 0 END) AS Age_0_10_female,
    SUM(CASE WHEN age_approx > 10 AND age_approx <= 25 AND sex = "male" THEN 1 ELSE 0 END) AS Age_10_25_male,
    SUM(CASE WHEN age_approx > 10 AND age_approx <= 25 AND sex = "female" THEN 1 ELSE 0 END) AS Age_10_25_female,
    SUM(CASE WHEN age_approx > 25 AND age_approx <= 40 AND sex = "male" THEN 1 ELSE 0 END) AS Age_25_40_male,
    SUM(CASE WHEN age_approx > 25 AND age_approx <= 40 AND sex = "female" THEN 1 ELSE 0 END) AS Age_25_40_female,
    SUM(CASE WHEN age_approx > 40 AND age_approx <= 60 AND sex = "male" THEN 1 ELSE 0 END) AS Age_40_60_male,
    SUM(CASE WHEN age_approx > 40 AND age_approx <= 60 AND sex = "female" THEN 1 ELSE 0 END) AS Age_40_60_female,
    SUM(CASE WHEN age_approx > 60 AND sex = "male" THEN 1 ELSE 0 END) AS Age_60_plus_male,
    SUM(CASE WHEN age_approx > 60 AND sex = "female" THEN 1 ELSE 0 END) AS Age_60_plus_female
FROM
    ISIC_DATA;


SELECT diagnosis, benign_malignant, COUNT(*) as image_count
FROM ISIC_DATA
GROUP BY diagnosis, benign_malignant
ORDER BY diagnosis, benign_malignant;