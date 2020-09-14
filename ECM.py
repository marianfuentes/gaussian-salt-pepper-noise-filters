import numpy as np

def ECM(ImgNoise, ImgFilter):
    if ImgNoise.shape == ImgFilter.shape:

        ECM = 0.0;
        row, col = ImgNoise.shape

        for x in range(row):
            for y in range(col):
                ECM = ECM + ( np.absolute(ImgNoise[x,y] - ImgFilter[x,y]) ** 2 )

        ECM = ECM / (col*row)
        return ECM

    else:
        raise InputError("Images must be the same size, returning")


