import random
import maya.cmds as cmds

cmds.commandPort(name=':7002',sourceType='python')

cmds.setAttr('blendShape4.Body',0.7)

LOOPS=1000

with open('f:/t7.txt','w') as t:
    x=[]
    w=0
    x.append(w)
    cmds.setAttr('Facial_Blends_nc11_2.Blink_Left',w)
    
