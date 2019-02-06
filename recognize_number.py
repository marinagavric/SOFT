# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 16:44:36 2019

@author: Marina
"""
import cv2
import numpy as np
import math

regioniZelena=[]
regioniPlava=[]
sviBrojeviZelena=[]
sviBrojeviPlava=[]
sviBrojevi=[]

class Broj:

    def __init__(self, x, y, dodat, oduzet, visina, sirina):
        self.x = x
        self.y = y
        self.dodat = dodat
        self.oduzet = oduzet
        self.visina = visina
        self.sirina = sirina
    def __repr__(self):
        return "<Broj x:%s y:%s dodat:%s oduzet:%s>" % (self.x, self.y, self.dodat, self.oduzet)

    def __str__(self):
        return "Broj: x is %s, y is %s" % (self.x, self.y) 
    

def checkRegion(linija, niz, broj,image_orig, image_bin, tip):
       levoX, veceY, desnoX, manjeY = linija.reshape(4)
       x, y, w, h = niz[0:4]
       
       if x >= levoX-w and x <= desnoX  and h>=12 and w>=1:
        vrednostFunkcije = getValue(linija,x,y)
        if y >= manjeY and y <= veceY:
               if vrednostFunkcije+h/2 <= 0:
                       if tip == 0:
                           if broj.dodat == 0:
                                broj.dodat = 1
                                region1 = image_bin[y:y + h+3 , x:x + w+3]
                                #cv2.imshow("katovana",region1)                       
                                kernel = np.ones((2,2),np.uint8)  
                                region1 = cv2.dilate(region1,kernel,iterations= 1)
                                region1 = cv2.erode(region1,kernel,iterations= 1)
                                region1= cv2.GaussianBlur(region1 ,(5,5),0)    
                                region1=cv2.resize(region1,(28,28), interpolation = cv2.INTER_NEAREST)
                                regioniPlava.append(region1)
                                cv2.imshow("plava",region1)
                                print('broj regiona plava')
                                print(len(regioniPlava))
                                #u rectangle prosledjujemo top-left tacku i bottom-right
                                cv2.rectangle(image_orig, (x, y), (x + w, y + h), (150, 255, 0), 2)
                       else:
                           if broj.oduzet == 0:
                                broj.oduzet = 1
                                region2 = image_bin[y:y + h+3 , x:x + w+3]
                               # cv2.imshow("katovana",region2)                       
                                kernel = np.ones((2,2),np.uint8)  
                                region2 = cv2.dilate(region2,kernel,iterations= 1)
                                region2 = cv2.erode(region2,kernel,iterations= 1)
                                region2= cv2.GaussianBlur(region2 ,(5,5),0)                        
                                region2=cv2.resize(region2,(28,28), interpolation = cv2.INTER_NEAREST)
                                regioniZelena.append(region2)
                                cv2.imshow("zelena",region2)
                                print('broj regiona zelena')
                                print(len(regioniZelena))
                                #u rectangle prosledjujemo top-left tacku i bottom-right
                                cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 255), 2)
                       
       return image_orig
    
def getValue(line,x,y):
            x1,y1,x2,y2 = line.reshape(4)
            
            k=(y2-y1)/(x2-x1)
            n=y1-k*x1
            
            vrednost=0
            
            vrednost =(k*x)+n-y
          #  print('vrednost jee')
            return vrednost

def recognizeNumbers(frame, greenLine, blueLine):
    img=frame;
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, image_bin = cv2.threshold(img_gray, 170, 255, cv2.THRESH_BINARY) 
    imgSend = np.copy(image_bin)
    img2, konture, hijerarhija = cv2.findContours(imgSend, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for kontura in konture:
        niz = []
        x, y, sirina, visina = cv2.boundingRect(kontura)
        niz.append(x)
        niz.append(y)
        niz.append(sirina)
        niz.append(visina)
        #print(niz)
        duzinaBrojeva = len(sviBrojevi)
        noviBroj = Broj(x, y, 0, 0, visina, sirina)
        noviBroj.dodat=0
        noviBroj.oduzet=0
           
        if(duzinaBrojeva != 0):
           # print('Ima brojeva')
            ubacen= 0
            for stariBroj in sviBrojevi:
                    #proveravamo da li je taj broj vec dodat u prethodnom frejmu
                    udaljenost = math.sqrt((noviBroj.x-stariBroj.x)*(noviBroj.x-stariBroj.x) + (noviBroj.y-stariBroj.y)*(noviBroj.y-stariBroj.y))
                    #print('udaljenost')
                    if udaljenost <= 25:
                        stariBroj.x = x
                        stariBroj.y = y
                        noviBroj= stariBroj
                        ubacen = 1
                        break
            
            if ubacen == 0:
                sviBrojevi.append(noviBroj)
                
        else:
            #print('Novi broj jer nema brojeva u nizu')
            sviBrojevi.append(noviBroj)
        #1 saljemo kad radimo sa zelenom linijom
        img=checkRegion(greenLine, niz, noviBroj,img, image_bin,1)
        img=checkRegion(blueLine, niz, noviBroj,img, image_bin,0)       
        cv2.imshow("video",img)
        #print(len(sviBrojevi))
     
     