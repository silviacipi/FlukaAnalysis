import cv2
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import ion
from scipy.signal.signaltools import wiener
import EllipseForKaz
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import math
import pylab as p
import matplotlib


from io import BytesIO

def selectData(v, nxsfileName,dataFolder):
    if v==1:
        print 'you selected HDF5 file'
        findContour(v, nxsfileName,dataFolder)
    else:
        print 'you selected TIF file'
        findContour(v, nxsfileName,dataFolder)
            
def findContour(nxsfileName,dataFolder):
    print nxsfileName
    
    mypath=h5py.File(nxsfileName,'r') 
    print 'looking for "',dataFolder, '" in the tree...'
    contLoop=True
    pathTot=''
    contLoop, pathToData, pathTot=myRec(mypath,contLoop,pathTot,dataFolder)
    print pathTot
    if not contLoop:
        print 'database "',dataFolder,'" found in  ', pathTot
        data=mypath[str(pathTot)]
        npdata=np.array(data)
        a,b,c=npdata.shape
        print a,b,c, ' file images to analyse' 
        #sizeA=int(math.sqrt(a))
        #print 'size', sizeA
        #s=(sizeA,sizeA)
        #image=np.zeros((b,c))
        imageorig=data[0][:][:]
        plt.imshow(imageorig, clim=[0,50])
        plt.show()
        
        aa=800
        bb=920
        #aa=810
        #bb=920
        #print np.shape()
        image=np.squeeze(data)[aa:bb,:]
        print np.shape(image)
        
        plt.imshow(image,clim=[0,50])
        plt.show()
        maximg=np.max(np.max(image))
        minimg=np.min(np.min(image))
        print 'maxvalue', maximg, minimg
        
        d,e=np.shape(image)     
        print d,e,'size'
        plt.figure(2)
        plt.imshow(image,clim=[0,50])
        plt.show()
        print 'this is d',d
        #position=np.zeros((1,2))
        '''
        colmin=1000
        colmax=3000
        rowmin=1450
        rowmax=1550
        threshold=350
        '''
        colmin=0
        colmax=2049
        rowmin=804
        rowmax=948
        threshold=1
        
        count=0
        for i in range(d):
            for j in range(0,2049):
                if image[i,j]>threshold:
                    #position[count][0]=i
                    #position[count][1]=j
                    count=count+1
        '''
        for i in range(d):
            for j in range(2701,e-500):
                if image[i][j]>threshold:
                    #position[count][0]=i
                    #position[count][1]=j
                    count=count+1
        '''
        positionX=np.zeros(count)
        positionY=np.zeros(count)
        count=0
        
        for i in range(d):
            for j in range(colmin,colmax):
                if image[i,j]>threshold:
                    positionX[count]=j
                    positionY[count]=i
                    print j,i
                    count=count+1
        '''
        for i in range(d):
            for j in range(2701,e-500):
                if image[i][j]>500:
                    positionX[count]=j
                    positionY[count]=i
                    count=count+1
        '''
        print positionX,positionY
        figure=plt.figure(10)
        plt.plot(positionX,positionY,'ro')
            
        #plt.show()
        #plt.figure(200)
        z=np.polyfit(positionX, positionY, 1)
        
        xrange=np.linspace(0, b, b)
        yrange=np.zeros(b)
        for i in range(b):
            yrange[i]=z[0]*xrange[i]+z[1]
        print np.shape(xrange)
        plt.plot(xrange,yrange,'r',linewidth=2.0)
        #plt.imshow(image)
        plt.show()
        #positionyCalc=
        
        print z
        '''
        print np.shape(image)
        counter=0
        minchan=int(channelmin/0.00125)
        maxchan=int(channelmax/0.00125)
        for i in range (sizeA):
#            print 'i', i
            for j in range (sizeA):
#               print j
                sum=0
                for col in range(minchan,maxchan):
                    sum=sum+npdata[counter][0][col]
                if i % 2 == 0:
#                print 'it is even '
                 image[j][i]=sum
#                print j,i
                else:
                 image[sizeA-1-j][i]=sum
#                print sizeA-1-j,i
                counter=counter+1
        fig1 = plt.figure(1)
        plt.imshow(image)
        plt.show()'''
    else:
        print 'database "', dataFolder,'" not found!'
    '''
        if v==1:
            a,b,c=npdata.shape
        else:
            a,b=npdata.shape
        print a, ' file images to analyse' 
        circlesProperties = np.zeros((1,2))
        fig3 = plt.figure(2)
        ax = fig3.gca()
        fig3.show()
        for i in range(a-1):
            print 'image ',i
            if v==1:
                # for HDF file
                img=npdata[i][:b][:c]
                blank_image = np.zeros((b,c,1), np.uint8)
            else:
                #For tif file
                #filename, file_extension = os.path.splitext(npdata[i][0])
                #########temporary
                mypath='C:\\Users\\xfz42935\\Documents\\Alignement\\64768-pco1-files' 
                #print mypath
                onlyfiles=[f for f in listdir(mypath) if isfile(join(mypath,f))]
                filename, file_extension = os.path.splitext(onlyfiles[i])             
                #########end of temporary
                if file_extension=='.tif':
                    #temporary
                    try:
                        #img=cv2.imread(join(mypath,onlyfiles[i]),cv2.IMREAD_UNCHANGED )
                        #end of temporary
                        img=cv2.imread(npdata[i][0],cv2.IMREAD_UNCHANGED )
                        height=np.size(img, 0)
                        width=np.size(img, 1)
                        blank_image = np.zeros((height,width,1), np.uint8)
                    except:
                        print 'image ',npdata[i][0]
                        print 'image not found: check the path'
                        continue
                else:
                    print 'tif image not found...looking for the next'
                    continue
                
            #minVal, maxVal, minLoc, maxLoc= cv2.minMaxLoc(img)   
            #temp=img/maxVal*255
            #blank_image=temp.astype(np.uint8)
            
            #### filtering
            img=wiener(img,mysize=9, noise=0.9)        
            height,width=img.shape
            blank_image = np.zeros((height,width,1), np.uint8)
            minVal, maxVal, minLoc, maxLoc= cv2.minMaxLoc(img)
            temp=img/maxVal*255
            #lowThresh=int((maxVal-minVal)/maxVal*255/4)
            #print 'low thresh',lowThresh
            blank_image=temp.astype(np.uint8)
            #ax.imshow(blank_image,cmap = 'gray')
            #fig3.canvas.draw()
            #ret, thresh=cv2.threshold(blank_image,50,255,cv2.THRESH_BINARY)  
            
            thresh=cv2.adaptiveThreshold(blank_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,31,0)   
            ax.imshow(thresh,cmap = 'gray')
            fig3.canvas.draw()                
            pippo=thresh.copy()
            #looking for contours
            pippo, contours,hierarchy = cv2.findContours(pippo, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            cnt=contours[0]
            centres = []
            test=0;
            for ii in range(len(contours)):
                moments = cv2.moments(contours[ii])
                  
                if not (moments['m00']==0):
                    p1=moments['m10']/moments['m00']
                    p2=moments['m01']/moments['m00']
                    if ii==1:
                        newrow = [[p1, p2]]
                        print 'circle found ', p1,p2
                        print 'hierarchy', hierarchy
                        centres.append([int(p1), int(p2)])
                       # print 'centre value', blank_image[int(p1), int(p2)]/maxVal, 'max value', maxVal
                        test=1
                        if i==0:
                            circlesProperties=newrow
                        else:
                            circlesProperties=np.vstack((circlesProperties,newrow))
                        break
            if not (test):
                print 'circle not found'        
        circlesProperties=np.array(circlesProperties)
        OneMat=np.ones_like(circlesProperties)    
        circlesProperties=circlesProperties+OneMat
        if len(circlesProperties)>1:
            y0, y1, y2, y3,y4=EllipseForKaz.Ellipse(circlesProperties[:,0], circlesProperties[:,1])
        else:
            print 'No circles found in images: check loaded nxs'
        print 'END'
        '''

      
def myRec(obj,continueLoop,pathTot,dataFolder):  
    ### recursive function to look for the data database
    temp=None
    i=1
    tempPath=''
    for name, value in obj.items():
        if continueLoop:
            #check if the object is a group
            if isinstance(obj[name], h5py.Group):
                tempPath='/'+name
                if len(obj[name])>0:
                    continueLoop,temp,tempPath= myRec(obj[name],continueLoop,tempPath,dataFolder)
                else:
                    continue
            else:
                test=obj[name]
                temp1='/'+dataFolder
                if temp1 in test.name:
                    continueLoop=False
                    tempPath=pathTot+'/'+name
                    return continueLoop,test.name,tempPath
            i=i+1
        if (i-1)>len(obj.items()):
            tempPath=''
    pathTot=pathTot+tempPath
    return continueLoop,temp, pathTot

    
   
#########For testing function
if __name__ == "__main__":
    pathToData='/home/xfz42935/Fluka/Bones/BonesTrabecularForImage3D-11102017-5.txt'
    
    matrix = np.loadtxt(pathToData)
    a,b=np.shape(matrix)
    print a,b
    xSteps=100
    ySteps=100
    zSteps=100
    
    newImage=np.zeros([xSteps,ySteps,zSteps])
    newImage2=np.zeros([xSteps*ySteps*zSteps,4])
    print 'image created'
    xCounter=0
    yCounter=0
    zCounter=0
    counter=0
    for i in range(a):
        for j in range(b):
            
            #if xCounter<100:
            print 'x,y,z', xCounter,yCounter,zCounter
            newImage[xCounter,yCounter,zCounter]=matrix[i,j]
            newImage2[counter,:]=[xCounter,yCounter,zCounter,matrix[i,j]]
            if xCounter<99:
                xCounter+=1
            else: 
                xCounter=0
                if yCounter<99:
                    yCounter+=1
                elif zCounter<99:
                    yCounter=0
                    zCounter+=1
            counter+=1
    
    #colors = (color_dimension  / color_dimension .max())
    #alph = color_dimension  / color_dimension .max()
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #color = np.asarray([(0,1,0, a) for a in (color_dimension/maxx)])
    
    newImage2[:,0]=newImage2[:,0]/10-5
    newImage2[:,1]=newImage2[:,1]/10-5
    newImage2[:,2]=newImage2[:,2]/10-5
    aver=np.average(newImage2[:,3])
    print 'average',aver
    item_b=newImage2[newImage2[:,2] <0]
    color_dimension = item_b[:,3]*1.602176462e-7*4.9e13*64e-3
    minn, maxx = color_dimension.min(), color_dimension.max()
    norm = matplotlib.colors.Normalize(minn, 9)
    m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    m.set_array([color_dimension])
    fcolors = m.to_rgba(color_dimension)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p=ax.scatter(item_b[:,0],item_b[:,2],item_b[:,1], c=fcolors,linewidth=0.0)#,clim=(0,10))
    ax.view_init(azim = 40)
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('z [cm]')
    ax.set_zlabel('y [cm]')
    
    max_range = np.array([item_b[:,0].max()-item_b[:,0].min(), item_b[:,2].max()-item_b[:,2].min(), item_b[:,1].max()-item_b[:,1].min()]).max() / 2.0

    mid_x = (item_b[:,0].max()+item_b[:,0].min()) * 0.5
    mid_y = (item_b[:,2].max()+item_b[:,2].min()) * 0.5
    mid_z = (item_b[:,1].max()+item_b[:,1].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    #ax.set_aspect('equal')
    
    cbar=plt.colorbar(m)
    cbar.set_ticks([0,3,6,9])
    cbar.set_ticklabels(['0','3','6','9'])
    cbar.set_label('Gy/64ms', rotation=270, x= 1,y=0.45)
    
    #png1 = BytesIO()
    #fig.savefig(png1, format='png')

    # (2) load this image into PIL
    dpi = 80
    fig.savefig('/home/xfz42935/Fluka/Bones/DoseMapZcut11102017-5PerProjection64mn.tif', dpi=dpi)

    # (3) save as TIFF
    #png2.save('3dPlot.tiff')
    #png1.close()
    
    
    #cbar.set_label_coords(1.05, -0.025)

    
    #fig.colorbar(p, shrink=0.85)
    #cbar.set_ytick( ticks=[0, 2,4,6,8])
    #cbar.set_yticklabels(['0', '4','6','8'])  # vertically oriented colorbar
    
    #ax.scatter(item_b[:,0],item_b[:,1],item_b[:,2], c=color_dimension,linewidth=0.0,clim=(0,10))
    #ax.scatter(newImage2[:,0],newImage2[:,1],newImage2[:,2], c=color,linewidth=0.0)#;)#,cmap='jet')#, alpha=alph)#
    ##ax.scatter(newImage2[:,0],newImage2[:,1],newImage2[:,2], c=color_dimension,linewidth=0.0)#;)#,cmap='jet')
    #ax.plot_surface(newImage2[:,0],newImage2[:,1],newImage2[:,2], rstride=1, cstride=1, facecolors=fcolors, vmin=minn, vmax=maxx, shade=False)
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('z')
    #fig.canvas.show()
    plt.show()
    print 'here'
    
    #colors = LikeBeta(newImage2[:,2],range(50),range(50))
    
    '''
    
    fig=p.figure()
    ax=fig.add_subplot(111, projection='3d')
    pmf = ax.plot_surface(newImage2[:,0], newImage2[:,1], newImage2[:,2], facecolors=cm.cmaps_listed   newImage2[:,3])
    #p.colorbar(pmf)
    p.show()    
    '''
    
    
    
    
    
    
    
    
    #, usecols=range(7))
    #pathToNexus='/dls/i13-1/data/2017/mt16702-2/raw/98141.nxs'
    #name='C:\\Users\\xfz4293newImage[:5\\Documents\\Alignement\\pco1-63429.hdf'
    #findContour(pathToNexus,'data')
