""" Same as 2 but with a spline function instead of joined linear fits.   """  

from Functions3 import *
random.seed(1)

"""Functions """

def calculateErrors(starlist, c, res):
    print "\n\nCalculating errors...\n\n"
    
    def callML(x):
        p=getP(x)
        return  LikelihoodFunction(starlist, p,c) 
    def makearray(value, nrows, ncols):
        return [[value]*ncols for _ in range(nrows)]
    cov = makearray(None,16,16)
    error=[]
    Hfun = nd.Hessian(callML)
    tempMatrix = matrix(  Hfun(res)  ) 
    print tempMatrix
    covarianceMatrix = np.array(tempMatrix.I).tolist()
    print "\nCovariance Matrix:\n", covarianceMatrix   
    
    for i in xrange(0,len(res),1):
        try: error.append( sqrt(-covarianceMatrix[i][i]))
        except: 
            error.append("Error")
            print "Hessian Error"         
    print "errors are:",error
    for i in xrange(0,len(res),1):
        for j in xrange(0,len(res),1):
            try:
                cov[i][j]= covarianceMatrix[i][j] / (error[i] * error[j])    
            except:
                print i,j
                cov[i][j]="Error"
    print "correlations are",cov    
    
            
def C(p,c,minColour,maxColour,spline,outputC):              # calculate normalisation coefficient    
    sigmaS, sigmaM =  getSigmaSSigmaMforC(p,minColour,maxColour)
    weight = getWeight(c,minColour,maxColour)
    
    def Ir(r0):
        def Ib(b0): 
            def Il(l0): return  exp((-0.5 * (p['R^2']  + r0 ** 2 - 2 * r0 * p['R'] *  cos(b0) *  cos(l0)) / (pow(sigmaS,2)) ))
            integralL = quad(Il, -0.04, 0.04)
            return integralL[0] *  cos(b0) 
        def Ibprp(bprp):  
            Mh = interpolate.splev(bprp,spline)
            m0= Mh + (5*log10(r0)) - 5 + AgCluster
            p = sigmaM*erfc(-((c['mlim']- m0)  /(sqrt(2)*sigmaM)))
            return p
        
        integralB = quad(Ib, -0.04, 0.04)
        integralBPRP= quad(Ibprp, minColour, maxColour)
        
        return integralB[0] * r0 ** 2 *  integralBPRP[0]
    integralR = quad(Ir,10, p['R'] + 100, epsabs =0)

    value =  0.5*(2*pi)**4 * p['sigmaMuAlpha'] * p['sigmaMuDelta']  * integralR[0] *weight #PUT WEIGHT BACK IN HERE !!!
    outputC.put(value)

def D(p,c,star,spline,outputstars,CsinEps):               # Calculate D for each star
    sigmaS, sigmaM =  getSigmaSSigmaMforD(p,star)
    thisCmin, thisCmax = getCminCmaxforD(star)
    weight = getWeight(c,thisCmin,thisCmax)

    def Imu(muAlpha0,muDelta0): return Fvel(star,muAlpha0,muDelta0,p)  * epsMuAlpha(star,muAlpha0) * epsMuDelta(star,muDelta0)
    integralMu = dblquad(Imu,-0.07,-0.03,lambda x: 0,lambda x: 0.05,epsabs=0)    
    
    def Ir(r0): return jacobian(star,r0) * Fspace(star,r0,p,sigmaS) * Fmag(star,r0,spline,sigmaM) * epsPi(star,r0) * integralMu[0]
    integralR = quad(Ir,10, p['R']+50)
    outputstars.put( weight *  integralR[0]  / (CsinEps * star['epsPi'] *star['epsMuAlpha'] * star['epsMuDelta']))  

def LikelihoodFunction(starlist, p, c):                # calculate the value of the likelihood functoin
    if checkParamsAreValid(p) == False: 
        print "The Sum: -10E307 \n"
        return -10E307  
    else:
        outputC, outputstars, procs, Plist = Queue(), Queue(), [], []
        spline = getSpline(bins,p)
        
        for colourRange in bins:
            procs = startProcess(procs, target=C, args=(p,c,colourRange[0],colourRange[1],spline,outputC))
        CsinEps = outputC.get()+outputC.get()+outputC.get()+outputC.get()
        for P in procs: P.join()
        procs=[]
        
        for star in starlist:
            star=transformCoordinates(star,c)
            procs = startProcess(procs, target=D, args=(p,c,star,spline,outputstars,CsinEps))
        
        for P in procs: P.join()
        
        
        for i in xrange (0,len(starlist),1):  Plist.append(  logD(outputstars.get(),star)  )
        finalSum = sum(Plist)
        print "The Sum:",finalSum,"\n"        
        return finalSum

def MLE(starlist, c, guess):
    t0 = time.time()
    print "\nMean coordinates are", c['lmean'], c['bmean']

    def f(x):      
        print x
        p = getP(x)
        return -LikelihoodFunction(starlist, p, c) 
    
    res =  minimize(f, x0=guess,method="Powell")
    print "NM: " , res.message, "\n", res.x, "\n", "Total time:", time.time() - t0
    return res.x   


""" Main program """    
if __name__ == '__main__':

    starlist = pd.read_csv("Pleiades", names=['mg', 'l', 'b', 'pi', 'epsPi',
                                              'muAlpha', 'epsMuAlpha', 'muDelta',
                                              'epsMuDelta', 'vr', 'epsVr', 'bp-rp']).T.to_dict().values()

    for star in starlist:
        star['sinl'] = sin(star['l'])
        star['cosl'] = cos(star['l'])
        star['sinb'] = sin(star['b'])
        star['cosb'] = cos(star['b'])
        star['x'] = (1.0 / star['pi']) * star['cosb'] * star['cosl']
        star['y'] = (1.0 / star['pi']) * star['cosb'] * star['sinl']
        star['z'] = (1.0 / star['pi']) * star['sinb']

    for i in range(len(starlist)):
        starlist[i].pop('vr', None)
        starlist[i].pop('epsVr-V', None)


    distlist,maxlist,minlist,llist,blist,xlist,ylist,zlist = [],[],[],[],[],[],[],[]

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
            
    meanx = np.mean(xlist)
    meany = np.mean(ylist)
    meanz = np.mean(zlist)
    print "mean cluster coordinates for original catalogue:",meanx,meany,meanz

    #results =  [ 46.35158593 , 3.41991717 ,  5.50180935 ,  2.29546342 ,  2.54500714,   0.76959647  , 3.98301261 ,  5.24713807 ,  7.05108992 ,  9.25966225,   0.39008765 ,  0.32244944,   0.63084539 , 0.34296494 ,0,0,10,10]
    results = [  1.25252094e+02 ,  3.35150575e+00,   4.86142031e+00 ,  1.03207132e+01,   1.31350468e+01 , -2.39384409e+00 ,  1.96855446e-01 ,  3.04079043e+00,   4.22140557e+00 ,  5.37379397e+00 ,  1.55576559e+00,   4.49615136e-01,   2.19895913e-01 ,  1.67554634e-01 ,  0,0,0.1,0.1]

    w=findW(starlist)

    print "Number of stars in cluster",len(starlist)
    c={'N':len(starlist),'Q':500,'mlim':12.0,'lmean':calculateMeanL(starlist, len(starlist)),'bmean':calculateMeanB(starlist, len(starlist)),"w":w}
    
    results = MLE(starlist, c, results)
    calculateErrors(starlist, c, results)
