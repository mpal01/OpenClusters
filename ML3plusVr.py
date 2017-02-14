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
    cov = makearray(None,len(res),len(res))
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
                print "i,j",i,j
    print "correlations are",cov    
                
def C(p,c,minColour,maxColour,spline,outputC):              # calculate normalisation coefficient    
    sigmaS, sigmaM =  getSigmaSSigmaMforC(p,minColour,maxColour)
    weight = getWeight(c,minColour,maxColour)
    
    def Ir(r0):
        def Ib(b0): 
            def Il(l0): return  exp((-0.5 * (p['R^2']  + r0 ** 2 - 2 * r0 * p['R'] *  cos(b0) *  cos(l0)) / (pow(sigmaS,2)) ))
            integralL = quad(Il, -0.1, 0.1)
            return integralL[0] *  cos(b0) 

        def Ibprp(bprp):  

            Mh = interpolate.splev(bprp,spline)
            m0= Mh + (5*log10(r0)) - 5 + AgCluster
            p = sigmaM*erfc(-((c['mlim']- m0)  /(sqrt(2)*sigmaM)))
            return p
        
        integralB = quad(Ib, -0.1, 0.1)
        integralBPRP= quad(Ibprp, minColour, maxColour)
        return integralB[0] * r0 ** 2 *  integralBPRP[0]
    integralR = quad(Ir,10, p['R'] + 100, epsabs =0)

    value =  (2*pi)**4*(1/sqrt(2)) * p['sigmaU'] * p['sigmaV'] * p['sigmaW'] * integralR[0] *weight
    outputC.put(value)

def D(p,c,star,spline,CsinEps):               # Calculate D for each star
    sigmaS, sigmaM =  getSigmaSSigmaMforD(p,star)
    thisCmin, thisCmax = getCminCmaxforD(star)
    weight = getWeight(c,thisCmin,thisCmax)
    
    def Ir(r0):  return jacobian(star,r0) * Fspace(star,r0,p,sigmaS) * Fmag(star,r0,spline,sigmaM)  * epsPi(star,r0) * Fv(p,star,r0)
    
    integralR = quad(Ir,0.1, p['R']+50) 
    thisD =  weight*integralR[0] / (CsinEps * star['epsPi'] * star['epsMuAlpha'] * star['epsMuDelta']) / sqrt(2 * pi) 
    return thisD

def processStars(chunk, outputStars, p, c,spline,CsinEps):       # Run D for a subset of the stars on one CPU
    templist, Dzero = [], 0
    for star in chunk:       
        star=transformCoordinates(star,c)
        temp=D(p,c,star,spline,CsinEps)
        templist.append(logD(temp,star))
        if logD(temp,star)==-10000:
            Dzero=Dzero+1 
    if Dzero > 0: print "Number of stars with Zero probability=",Dzero
    thesum = sum(templist)            
    outputStars.put(thesum)

def LikelihoodFunction(starlist, p,c):                # calculate the value of the likelihood functoin
    
    if checkParamsAreValid(p) == False: 
        return -10E307  
    else:
        t0 = time.time()
        output, outputC, outputstars, procs, Plist = 0, Queue(), Queue(), [], []
        spline = getSpline(bins,p)
        t0c = time.time()
        for colourRange in bins:
            procs = startProcess(procs, target=C, args=(p,c,colourRange[0],colourRange[1],spline,outputC))
            
        for P in procs: P.join()
        #print "time for C:",time.time()-t0c
        CsinEps = outputC.get()+outputC.get()+outputC.get()+outputC.get()
        procs=[]
        
        t0d = time.time()
        chunks = [starlist[x:x + c['Q']] for x in xrange(0, len(starlist), c['Q'])]
        for chunk in chunks: procs = startProcess(procs, target=processStars, args=(chunk, outputstars, p, c,spline,CsinEps))
        
        for P in procs: P.join()
        #print "time for D:",time.time()-t0d
        for item in chunks: output = output + outputstars.get() 
        print "time total:",time.time()-t0     
        print "The Sum:",output,"\n"   
        
        return output

def MLE(starlist, c, guess):
    t0 = time.time()
    print "\nMean coordinates are", c['lmean'], c['bmean']

    def f(x):      
        print x
        p = getP(x)
        time.sleep(0.5)
        return -LikelihoodFunction(starlist, p, c) 
    
    res =  minimize(f, x0=guess,method="Powell")  
    print "NM: " , res.message, "\n", res.x, "\n", "Total time:", time.time() - t0
    return res.x   


""" Main program """    
if __name__ == '__main__':
    #results = [      121.11351406 ,  11.46465626,   16.77702542 ,  27.64636401,   31.63797537 ,  -2.48363872 ,   0.17972312 ,   2.97634413  ,  4.15402883,    5.30213511 ,   2.33864906 ,   0.73859052 ,   0.54900779  ,  0.53032651,   -6.9923114 ,  -25.76952654,  -13.64454163, 5.8516476] 
    results =  [ 46.35158593 , 3.41991717 ,  5.50180935 ,  2.29546342 ,  2.54500714,   0.76959647  , 3.98301261 ,  5.24713807 ,  7.05108992 ,  9.25966225,   0.39008765 ,  0.32244944,   0.63084539 , 0.34296494 ,-42.24407887, -19.27759118 , -1.54856755,   1.09887813] 

    starlist = makeSample()  
    w=findW(starlist)
    
    """
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
    #plt.xlim([25,65])
    #plt.scatter(ylist,zlist)
    distancetocenterlist=[]
    for star in starlist:
        distancetocenter = sqrt( abs(star['x']-meanx)**2 + abs(star['y']-meany)**2 + abs(star['z']-meanz)**2 )
        star['distancetocenter'] = distancetocenter
    #plt.errorbar(distlist,llist, xerr=[maxlist,minlist], fmt='b', ls='None')
    
    """
    
    """plt.hist(distancetocenterlist)
    plt.savefig("HyadesDistancetoCenter")
    sys.exit()
    """
    
    """
    newstarlist=[]
    for star in starlist:
        if star['distancetocenter']<10.0: newstarlist.append(star)
    starlist=newstarlist
    
    
    for star in starlist: 
        plt.scatter(star['bp-rp'],star['obsMg'])
    plt.savefig("HyadesHRInner10")
    print results
    """
    
    print "Number of stars in cluster",len(starlist)
    c={'N':len(starlist),'Q':10,'mlim':12.0,'lmean':calculateMeanL(starlist, len(starlist)),'bmean':calculateMeanB(starlist, len(starlist)),"w":w} #,'rmean':calculateMeanR(starlist,len(starlist))}
    
    
    """
    vrlist = [star['vr'] for star in starlist if star['epsVr']<50]
    epslist = [star['epsVr'] for star in starlist if star['epsVr']<50]
    
    sdev = np.std(vrlist)
    meaneps = np.mean(epslist)

    dispersion = sqrt(  sdev**2 - meaneps**2 )
    print "the dispersion is", dispersion 
    
    vrlist = [star['muAlpha'] for star in starlist]
    epslist = [star['epsMuAlpha'] for star in starlist]
    
    sdev = np.std(vrlist)
    meaneps = np.mean(epslist)

    dispersion = sqrt(  sdev**2 - meaneps**2 )
    print "the dispersion is", dispersion 
    
    final = 46*sin(dispersion / 3600000 )   * 10**6 
    print final
    sys.exit()
    """
    
    #starlist, noneRemoved = removeOutliers(starlist,results,c)
    #w=findW(starlist)
    #c.update({"w":w})
    print "remaining stars in sample:",len(starlist)
    
       
    
    #c.update({'N':len(starlist)})
    results =MLE(starlist, c, results)
    calculateErrors(starlist, c, results)   
                
    
