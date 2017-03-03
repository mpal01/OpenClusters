"""
@author: M. Palmer

Computes the ML estimates of the parameters of the model being the stars belonging to a cluster which is spherical at
some distance, with the absolute magnitude following a isochrone and moving trough the space with some velocity.

Input: a file with a list of cluster members with the following parameters:
magnitud G        = mg (mag)
galatic latitude  = l (radians)
galatic longitude = b (radians)
parallax          = pi (muas)
parallax error    = epsPi (muas)
mualpha           = muAlpha (muas)
mualpha error     = epsMuAlpha (muas)
mudelta           = muDelta (muas)
mudelta error     = epsMuDelta (muas)
color             = bp-rp (mag) (bt-vt can be used)

Output:
Parameters, distance, position, speed, size isochrone, with their errors

"""
from Functions3 import *
random.seed(1)

def calculateErrors(starlist, c, res):
    '''
    Finds the derivative of the likelihood function around the maximum computing a Hessian matrix
    (Please refer to M. Palmer et al (2014)

    :param starlist: star list
    :param c: star dictionary with number of stars, Gaia mag limit, lMean, bMean, w
    :param res: MLE results
    :return: covariance matrix
    '''

    print "\n\nCalculating errors...\n\n"
    
    def callML(x):
        p=getP(x)
        return  LikelihoodFunction(starlist, p,c)

    def makearray(value, nrows, ncols):
        return [[value]*ncols for _ in range(nrows)]

    cov = makearray(None,16,16)
    error=[]
    Hfun = nd.Hessian(callML)
    tempMatrix = matrix(Hfun(res))

    print tempMatrix
    covarianceMatrix = np.array(tempMatrix.I).tolist()
    print "\nCovariance Matrix:\n", covarianceMatrix   
    
    for i in xrange(0,len(res),1):
        try: error.append( sqrt(-covarianceMatrix[i][i]))
        except: 
            error.append("Error")
            print "Hessian Error"         
    print "Errors are:",error


    for i in xrange(0,len(res),1):
        for j in xrange(0,len(res),1):
            try:
                cov[i][j]= covarianceMatrix[i][j] / (error[i] * error[j])    
            except:
                print i,j
                cov[i][j]="Error"
    print "correlations are",cov    
    
            
def calculatesNormalizationCoeff(p, c, minColour, maxColour, spline, outputC):
    '''
    Calculates the normalisation coefficient
    :param p:
    :param c:
    :param minColour:
    :param maxColour:
    :param spline:
    :param outputC:
    :return: nothing
    '''
    sigmaS, sigmaM =  getSigmaSSigmaMforC(p,minColour,maxColour)
    weight = getWeight(c,minColour,maxColour)

    mlim = c['mlim']
    Rsquared = p['R^2']
    R = p['R']
    sigmaSsquared = pow(sigmaS,2)
    delimiter = sqrt(2)*sigmaM

    def Ir(r0):
        def Ib(b0):
            def Il(l0): return exp(-0.5 * (Rsquared + r0 ** 2 - 2 * r0 * R *  cos(b0) * cos(l0)) / sigmaSsquared)
            integralL = quad(Il, -0.04, 0.04)
            return integralL[0] *  cos(b0)
        def Ibprp(bprp):  
            Mh = interpolate.splev(bprp,spline)
            m0= Mh + (5*log10(r0)) - 5 + AgCluster
            p = sigmaM*erfc(-((mlim - m0) / delimiter))
            return p
        
        integralB = quad(Ib, -0.04, 0.04)
        integralBPRP= quad(Ibprp, minColour, maxColour)
        
        return integralB[0] * r0 ** 2 *  integralBPRP[0]
    integralR = quad(Ir,10, p['R'] + 100, epsabs =0)


    value =  0.5*(2*pi)**4 * p['sigmaMuAlpha'] * p['sigmaMuDelta']  * integralR[0] *weight #PUT WEIGHT BACK IN HERE !!!
    outputC.put(value)

def calculatesUnnormaLikelihood(p, c, star, spline, outputstars, CsinEps):
    '''
    Calculates the unnormalized Likelihood for each star
    :param p:
    :param c:
    :param star:
    :param spline:
    :param outputstars:
    :param CsinEps:
    :return:
    '''
    sigmaS, sigmaM =  getSigmaSSigmaMforD(p,star)
    thisCmin, thisCmax = getCminCmaxforD(star)
    weight = getWeight(c,thisCmin,thisCmax)




    def Imu(muAlpha0,muDelta0): return Fvel(star,muAlpha0,muDelta0,p)  * epsMuAlpha(star,muAlpha0) * epsMuDelta(star,muDelta0)
    integralMu = dblquad(Imu,-0.07,-0.03,lambda x: 0,lambda x: 0.05,epsabs=0)    
    
    def Ir(r0): return jacobian(star,r0) * Fspace(star,r0,p,sigmaS) * Fmag(star,r0,spline,sigmaM) * epsPi(star,r0) * integralMu[0]
    integralR = quad(Ir,10, p['R']+50)


    outputstars.put( weight *  integralR[0]  / (CsinEps * star['epsPi'] *star['epsMuAlpha'] * star['epsMuDelta']))

def LikelihoodFunction(starlist, p, c):
    '''
    Calculates the value of the likelihood functin for every star
    :param starlist:
    :param p:
    :param c:
    :return:
    '''
    if checkParamsAreValid(p) == False: 
        print "The Sum: -10E307 \n"
        return -10E307  
    else:
        outputC, outputstars, procs, Plist = Queue(), Queue(), [], []
        spline = getSpline(bins,p)

        t0 = time.time()

        for colourRange in bins:
            procs = startProcess(procs, target=calculatesNormalizationCoeff, args=(p, c, colourRange[0], colourRange[1], spline, outputC))
        CsinEps = outputC.get()+outputC.get()+outputC.get()+outputC.get()
        for P in procs: P.join()
        procs=[]

        print time.time() - t0

        for star in starlist:
            star=transformCoordinates(star,c)
            procs = startProcess(procs, target=calculatesUnnormaLikelihood, args=(p, c, star, spline, outputstars, CsinEps))
        
        for P in procs: P.join()

        
        for i in xrange (0,len(starlist),1):  Plist.append(  logD(outputstars.get(),star)  )
        finalSum = sum(Plist)
        print "The Sum:",finalSum,"\n"        
        return finalSum


def MLE(starlist, c, guess):
    '''
    MLE wrapper

    :param starlist:
    :param c:
    :param guess:
    :return:
    '''
    t0 = time.time()
    print "\nMean coordinates are", c['lmean'], c['bmean']

    def f(x):      
        print x
        p = getP(x)
        return -LikelihoodFunction(starlist, p, c) 
    
    res =  minimize(f, x0=guess,method="Powell")
    print "NM: " , res.message, "\n", res.x, "\n", "Total time:", time.time() - t0
    return res.x   


"""

Main program to calculate ML. Catalogue should be located inside the same directory

"""
if __name__ == '__main__':


    # Reads a csv file containing the parameters
    fileName = 'Pleiades'
    starlist = pd.read_csv(fileName, names=['mg', 'l', 'b', 'pi', 'epsPi',
                                              'muAlpha', 'epsMuAlpha', 'muDelta',
                                              'epsMuDelta', 'bp-rp']).T.to_dict().values()

    # Coordinate transformation to 3 dimensional space x, y and z
    for star in starlist:
        star['sinl'] = sin(star['l'])
        star['cosl'] = cos(star['l'])
        star['sinb'] = sin(star['b'])
        star['cosb'] = cos(star['b'])
        star['x'] = (1.0 / star['pi']) * star['cosb'] * star['cosl']
        star['y'] = (1.0 / star['pi']) * star['cosb'] * star['sinl']
        star['z'] = (1.0 / star['pi']) * star['sinb']

    # In case of having radial velocities
    #for i in range(len(starlist)):
    #    starlist[i].pop('vr', None)
    #    starlist[i].pop('epsVr-V', None)


    distlist,maxlist,minlist,llist,blist,xlist,ylist,zlist = [],[],[],[],[],[],[],[]

    # Creates list of values for each parameter
    for star in starlist:
        if star['pi']>0: 
            distlist.append(1/star['pi'])
            llist.append(star['l'])
            blist.append(star['b'])
            maxlist.append((1/star['pi'])-(1/(star['pi']-star["epsPi"])))
            minlist.append(-((1/star['pi'])-(1/(star['pi']+star["epsPi"]))))
            xlist.append(star['x'])
            ylist.append(star['y'])
            zlist.append(star['z'])

    # x, y and z means
    meanx = np.mean(xlist)
    meany = np.mean(ylist)
    meanz = np.mean(zlist)
    print "mean cluster coordinates for original catalogue:",meanx,meany,meanz

    # Initial Guess for Hyades
    #initial_guess =  [ 46.35158593 , 3.41991717 ,  5.50180935 ,  2.29546342 ,  2.54500714,   0.76959647  , 3.98301261 ,  5.24713807 ,  7.05108992 ,  9.25966225,   0.39008765 ,  0.32244944,   0.63084539 , 0.34296494 ,0,0,10,10]

    # Initial Guess for Pleiades
    initial_guess = [  1.25252094e+02 ,  3.35150575e+00,   4.86142031e+00 ,  1.03207132e+01,   1.31350468e+01 , -2.39384409e+00 ,  1.96855446e-01 ,  3.04079043e+00,   4.22140557e+00 ,  5.37379397e+00 ,  1.55576559e+00,   4.49615136e-01,   2.19895913e-01 ,  1.67554634e-01 ,  0,0,0.1,0.1]

    # Find number of stars in every bin (color bin)
    w=findW(starlist)

    print "Number of stars in cluster",len(starlist)
    c={'N':len(starlist),'Q':500,'mlim':20.0,'lmean':calculateMeanL(starlist, len(starlist)),'bmean':calculateMeanB(starlist, len(starlist)),"w":w}

    # Calculates ML and errors
    results = MLE(starlist, c, initial_guess)
    calculateErrors(starlist, c, results)
