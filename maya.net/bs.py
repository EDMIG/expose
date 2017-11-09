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
    y=cmds.pointPosition('Body.vtx[6277]',w=1);x.append(y)
    y=cmds.pointPosition('Body.vtx[4131]',w=1);x.append(y)
    y=cmds.pointPosition('Body.vtx[4133]',w=1);x.append(y)
    y=cmds.pointPosition('Body.vtx[4143]',w=1);x.append(y)
    y=cmds.pointPosition('Body.vtx[3402]',w=1);x.append(y)
    y=cmds.pointPosition('Body.vtx[3496]',w=1);x.append(y)
    y=cmds.pointPosition('Body.vtx[3550]',w=1);x.append(y)
    y=cmds.pointPosition('Body.vtx[3504]',w=1);x.append(y)

    t.write(str(x)+'\n')

    for i in range(1,LOOPS):

        x=[]
        w=random.random()
        x.append(w)
        cmds.setAttr('Facial_Blends_nc11_2.Blink_Left',w)
        y=cmds.pointPosition('Body.vtx[6277]',w=1);x.append(y)
        y=cmds.pointPosition('Body.vtx[4131]',w=1);x.append(y)
        y=cmds.pointPosition('Body.vtx[4133]',w=1);x.append(y)
        y=cmds.pointPosition('Body.vtx[4143]',w=1);x.append(y)
        y=cmds.pointPosition('Body.vtx[3402]',w=1);x.append(y)
        y=cmds.pointPosition('Body.vtx[3496]',w=1);x.append(y)
        y=cmds.pointPosition('Body.vtx[3550]',w=1);x.append(y)
        y=cmds.pointPosition('Body.vtx[3504]',w=1);x.append(y)

        t.write(str(x)+'\n')
