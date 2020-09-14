# Pontificia Universidad Javeriana. Departamento de Electrónica
# Authors: Juan Henao, Marian Fuentes; Estudiantes de Ing. Electrónica.
# Procesamiento de Imagenes y video
# 14/09/2020

# Importing Librarys
import numpy as np
import cv2
import os
from noise import * # Provided By Julian Quiroga
import time
from ECM import * # Made by group to calculate EMC

# Main Code
if __name__ == "__main__":
    #############################################
    # 1. Read lena and create noisy lena Images #
    #############################################
    path = "C:/Users/ACER/Desktop/Semestre10/Imagenes/Presentaciones/Semana 6/Imagenes"
    name = "lena.png"
    path_name = os.path.join(path, name) #Join path and name
    lena = cv2.imread(path_name) #Read Image
    lena = cv2.cvtColor(lena, cv2.COLOR_BGR2GRAY) #Change lena to Gray CS
    #cv2.imshow("lena", lena) #If you want to see the Image.

    ###########################################################################################
    # 1.1 Generate lena_gauss_noisy & lena_s&p_noisy using noise() provided by Julian Quiroga #
    ###########################################################################################
    lena_gauss_noisy = noise("gauss", lena.astype(np.float) / 255) #Generate gaussian noisy lena
    lena_gauss_noisy = (255 * lena_gauss_noisy).astype(np.uint8)

    lena_sp_noisy = noise("s&p", lena.astype(np.float) / 255)
    lena_sp_noisy = (255 * lena_sp_noisy).astype(np.uint8) # Generate Salt and pepper noisy lena

    #cv2.imshow("lena gauss noise", lena_gauss_noisy) #If you want to see the Image.
    #cv2.imshow("lena s&p noise", lena_sp_noisy) #If you want to see the Image.

    #################################
    # 1.2 Filter Noisy lena´s with: #
    #################################

    ###########################################
    # 1.2.1 Gaussian low-pass 7x7 sigma = 1.5 #
    ###########################################
    N = 7
    start = time.perf_counter() #Measuring filter implementation time in s
    lenaGSN_GLP = cv2.GaussianBlur(lena_gauss_noisy, (N, N), 1.5, 1.5) # apply filter
    stop = time.perf_counter() #Measuring filter implementation time in s
    GLP_Ft1 = stop - start #Measuring filter implementation time in s

    start = time.perf_counter() # Measuring filter implementation time in s
    lenaSPN_GLP = cv2.GaussianBlur(lena_sp_noisy, (N, N), 1.5, 1.5) # apply filter
    stop = time.perf_counter()  # Measuring filter implementation time in s
    GLP_Ft2 = stop - start # Measuring filter implementation time in s

    #cv2.imshow("GSN / Gauss LP filter", lenaGSN_GLP) #If you want to see the Image.
    #cv2.imshow("SPN / Gauss LP filter", lenaSPN_GLP) #If you want to see the Image.

    ###########################
    # 1.2.2 Median Filter 7x7 #
    ###########################

    start = time.perf_counter()  # Measuring filter implementation time in s
    lenaGSN_median = cv2.medianBlur(lena_gauss_noisy, 7) # apply filter
    stop = time.perf_counter()  # Measuring filter implementation time in s
    mediant1 = stop - start # Measuring filter implementation time in s

    start = time.perf_counter()  # Measuring filter implementation time in s
    lenaSPN_median = cv2.medianBlur(lena_sp_noisy, 7) # apply filter
    stop = time.perf_counter()  # Measuring filter implementation time in s
    mediant2 = stop - start # Measuring filter implementation time in s

    #cv2.imshow("lena GSN / Median", lenaGSN_median) #If you want to see the Image.
    #cv2.imshow("lena SPN / Median", lenaSPN_median) #If you want to see the Image.

    ################################################################
    ## 1.2.3 Bilateral filter d=15, sigmaColor = sigmaSpace = 25. ##
    ################################################################

    start = time.perf_counter()  # Measuring filter implementation time in s
    lenaGSN_bilateral = cv2.bilateralFilter(lena_gauss_noisy, 15, 25, 25) # apply filter
    stop = time.perf_counter()  # Measuring filter implementation time in s
    bilateralt1 = stop - start # Measuring filter implementation time in s

    start = time.perf_counter()  # Measuring filter implementation time in s
    lenaSPN_bilateral = cv2.bilateralFilter(lena_sp_noisy, 15, 25, 25) # apply filter
    stop = time.perf_counter()  # Measuring filter implementation time in s
    bilateralt2 = stop - start # Measuring filter implementation time in s

    #cv2.imshow("lena GN Bilateral", lenaGSN_bilateral) #If you want to see the Image.
    #cv2.imshow("lena GN Bilateral", lenaSPN_bilateral) #If you want to see the Image.

    ##########################################################################
    # 1.2.4 Non Local Means NLM Filter h=5, windowSize = 15, searchSize = 25 #
    ##########################################################################
    start = time.perf_counter()  # Measuring filter implementation time in s
    lenaGSN_NLM = cv2.fastNlMeansDenoising(lena_gauss_noisy, 5, 15, 25) # apply filter
    stop = time.perf_counter()  # Measuring filter implementation time in s
    NLM_Ft1 = stop - start # Measuring filter implementation time in s

    start = time.perf_counter()  # Measuring filter implementation time in s
    lenaSPN_NLM = cv2.fastNlMeansDenoising(lena_sp_noisy, 5, 15, 25) # apply filter
    stop = time.perf_counter() # Measuring filter implementation time in s
    NLM_Ft2 = stop - start # Measuring filter implementation time in s

    #cv2.imshow("GN / NLM FILTER", lenaGSN_NLM) # if you want to see the Image
    #cv2.imshow("S&P / NLM FILTER", lenaSPN_NLM) # if you want to see the Image

    ##############################################################
    # 1.3 Noise estimation for every noisy-filtered image pair #
    ##############################################################

    ###########################
    # Gaussian Low Pass Filter#
    ###########################

    lenaNE_GSN_GLP = np.absolute(lena_gauss_noisy - lenaGSN_GLP) # Calculate Noise estimation
    lenaNE_SPN_GLP = np.absolute(lena_sp_noisy - lenaSPN_GLP) # Calculate Noise estimation

    #cv2.imshow("Noise Estimation GSN/GLP ", lenaNE_GSN_GLP) # if you want to see the image
    #cv2.imshow("Noise Estimation SPN/GLP ", lenaNE_SPN_GLP) # if you want to see the image

    ###########################
    # Median Filter 7x7 #######
    ###########################

    lenaNE_GSN_median = np.absolute(lena_gauss_noisy - lenaGSN_median) # Calculate Noise estimation
    lenaNE_SPN_median = np.absolute(lena_sp_noisy - lenaSPN_median) # Calculate Noise estimation

    # cv2.imshow("Noise Estimation GSN/median ", lenaNE_GSN_median) # if you want to see the image
    # cv2.imshow("Noise Estimation SPN/median ", lenaNE_SPN_median) # if you want to see the image

    ################################################################
    ## Bilateral filter d=15, sigmaColor = sigmaSpace = 25. ########
    ################################################################

    lenaNE_GSN_bilateral = np.absolute(lena_gauss_noisy - lenaGSN_bilateral) # Calculate Noise estimation
    lenaNE_SPN_bilateral = np.absolute(lena_sp_noisy - lenaSPN_bilateral) # Calculate Noise estimation

    #cv2.imshow("Estimation: Gaussian noise, Bilateral filter ", lenaNE_GSN_bilateral)
    #cv2.imshow("Estimation Salt & Pepper noise, Bilateral filter", lenaNE_SPN_bilateral)

    ##########################################################################
    # Non Local Means NLM Filter h=5, windowSize = 15, searchSize = 25 #######
    ##########################################################################

    lenaNE_GSN_NLM = np.absolute(lena_gauss_noisy - lenaGSN_NLM) # Calculate Noise estimation
    lenaNE_SPN_NLM = np.absolute(lena_sp_noisy - lenaSPN_NLM ) # Calculate Noise estimation

    #cv2.imshow("Noise Estimation GSN/NLM ", lenaNE_GSN_NLM) # if you want to see the image
    #cv2.imshow("Noise Estimation SPN/NLM ", lenaNE_SPN_NLM) # if you want to see the image

    #######################################################################
    # 2 find Mean Squared Error, noted in code as ECM (in spanish :p) #####
    #######################################################################

    ###########################
    # Gaussian Low Pass Filter#
    ###########################

    ECM_GSN_GLP = ECM(lena.astype(np.float) / 255, lenaGSN_GLP.astype(np.float) / 255) # Calculate ECM using ECM def
    ECM_GSN_GLP = ECM_GSN_GLP * 255 # Calculate ECM using ECM def
    print("Mean Squared Error GSN / GLP : ", ECM_GSN_GLP)

    ECM_SPN_GLP = ECM(lena.astype(np.float) / 255, lenaSPN_GLP.astype(np.float) / 255) # Calculate ECM using ECM def
    ECM_SPN_GLP = ECM_SPN_GLP * 255 # Calculate ECM using ECM def
    print("Mean Squared Error SPN / GLP : ", ECM_SPN_GLP)


    ###########################
    # Median Filter 7x7 #######
    ###########################

    ECM_GSN_median = ECM(lena.astype(np.float) / 255, lenaGSN_median.astype(np.float) / 255) # Calculate ECM using ECM def
    ECM_GSN_median = ECM_GSN_median*255 # Calculate ECM using ECM def
    print("Mean Squared Error GSN / median : ", ECM_GSN_median)

    ECM_SPN_median = ECM(lena.astype(np.float) / 255, lenaSPN_median.astype(np.float) / 255) # Calculate ECM using ECM def
    ECM_SPN_median = ECM_SPN_median * 255 # Calculate ECM using ECM def
    print("Mean Squared Error SPN / median : ", ECM_SPN_median)

    ################################################################
    ## Bilateral filter d=15, sigmaColor = sigmaSpace = 25. ########
    ################################################################

    ECM_GSN_bilateral = ECM(lena.astype(np.float) / 255, lenaGSN_bilateral.astype(np.float) / 255)
    ECM_GSN_bilateral = ECM_GSN_bilateral * 255 # Calculate ECM using ECM def
    print("Mean Squared Error GSN / bilateral : ", ECM_GSN_bilateral)

    ECM_SPN_bilateral = ECM(lena.astype(np.float) / 255, lenaSPN_bilateral.astype(np.float) / 255)
    ECM_SPN_bilateral = ECM_SPN_bilateral * 255 # Calculate ECM using ECM def
    print("Mean Squared Error SPN / bilateral : ", ECM_SPN_bilateral)

    ##########################################################################
    # Non Local Means NLM Filter h=5, windowSize = 15, searchSize = 25 #######
    ##########################################################################

    ECM_GSN_NLM = ECM(lena.astype(np.float) / 255, lenaGSN_NLM.astype(np.float) / 255)
    ECM_GSN_NLM = ECM_GSN_NLM * 255 # Calculate ECM using ECM def
    print("Mean Squared Error GSN / NLM : ", ECM_GSN_NLM)

    ECM_SPN_NLM = ECM(lena.astype(np.float) / 255, lenaSPN_NLM.astype(np.float) / 255)
    ECM_SPM_NLM = ECM_SPN_NLM * 255 # Calculate ECM using ECM def
    print("Mean Squared Error SPN / NLM : ", ECM_SPN_NLM)

    #######################################################################
    # 3 Calculating execution time for different filters ##################
    #######################################################################

    ###########################
    # Gaussian Low Pass Filter#
    ###########################

    TimeGLP = 0.5*(GLP_Ft1+GLP_Ft2) #Calculate time execution
    print("Gaussian Low Pass Filter time execution is: ", TimeGLP, "s")

    ###########################
    # Median Filter 7x7 #######
    ###########################

    TimeMedian = 0.5*(mediant1+mediant2) #Calculate time execution
    print("Median filter time execution is: ", TimeMedian, "s")

    ################################################################
    ## Bilateral filter d=15, sigmaColor = sigmaSpace = 25. ########
    ################################################################

    TimeBilateral = 0.5*(bilateralt1+bilateralt2) #Calculate time execution
    print("Bilateral filter time execution is: ", TimeBilateral, "s")

    ##########################################################################
    # Non Local Means NLM Filter h=5, windowSize = 15, searchSize = 25 #######
    ##########################################################################

    TimeNLM = 0.5*(NLM_Ft1+NLM_Ft2) #Calculate time execution
    print("Non Local Means Time execution is : ", TimeNLM, "s")

    #cv2.waitKey(0) # #--->>> If you are going to make visible ANY image Uncomment this first !

##############################################################################################################
# Pontificia Universidad Javeriana, Sede Bogota. #############################################################
# Authors: Juan Henao & Marian Fuentes. - Proc. de imagenes y video ##########################################
# ############################################################################################################