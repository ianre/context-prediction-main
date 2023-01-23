import os, sys
from pipeline_scripts.mask_context_evaluation import Context_Iterator
from pipeline_scripts.contour_extraction import Contour_Iterator
from pipeline_scripts.metrics import Metrics_Iterator
import pipeline_scripts.utils
import time

def getTrialFrames(cwd,task,trial):
    TrialRoot = os.path.join(cwd,task,"images",trial)
    for root, dirs, files in os.walk(TrialRoot):
        Frames = files
        break    
    
    frameNumbers = []
    for f in Frames:
        frameNumber = f.replace(".png","").split("_")[1]
        frameNumbers.append(frameNumber)
    #print("Frames",Frames,"\nFNums",frameNumbers)
    #print("FNums",frameNumbers)
    return frameNumbers

def main():
    CWD=os.getcwd()
    TASK = "Knot_Tying" #DEFAULT 
    try:
        TASK=sys.argv[1]
    except:
        print("Error: no task provided","Usage: python context_gen_pipeline.py <task>","Default Task"+TASK)
    taskDir = os.path.join(CWD, TASK,"images")
    trials = [] 
    for root, dirs, files in os.walk(taskDir):
        trials = dirs
        break
        
    
    BATCH_SIZE = 10
    EPOCH = 0
    TRIAL = trials[0]
    TRIAL_FRAMES = getTrialFrames(CWD,TASK,TRIAL) # formatted as the XXXX in 'frame_XXXX.png'
    

    print("Pipeline Running for task ",TASK,"trial",TRIAL)
    while(EPOCH * BATCH_SIZE < len(TRIAL_FRAMES)):
        print("EPOCH ", EPOCH, ":", TRIAL_FRAMES[BATCH_SIZE*EPOCH:BATCH_SIZE*EPOCH+BATCH_SIZE])
        start_time = time.time()
        I = Contour_Iterator(TASK,CWD)
        label_classes, label_classNames, ContourFiles = I.ExtractContours(BATCH_SIZE,EPOCH,TRIAL,TRIAL_FRAMES[BATCH_SIZE*EPOCH:BATCH_SIZE*EPOCH+BATCH_SIZE])
        I = Context_Iterator(TASK,CWD)
        #I.GenerateContext(BATCH_SIZE,EPOCH,TRIAL,TRIAL_FRAMES[BATCH_SIZE*EPOCH:BATCH_SIZE*EPOCH+BATCH_SIZE],SAVE=True)
        end_time = time.time()
        print(" %s s \n" % round((end_time - start_time),3), end="")
        EPOCH +=1

    I = Metrics_Iterator(TASK,CWD)
    I.IOU()
    return

    #start
    start_time = time.time()
    print("Pipeline Running for task ",TASK)
    print("Contour Extraction")
    I = Contour_Iterator(TASK,CWD)
    I.ExtractContours(BATCH_SIZE,EPOCH,TRIAL_FRAMES[BATCH_SIZE*EPOCH:BATCH_SIZE*EPOCH+BATCH_SIZE])
    #label_classes, label_classNames = getLabelClassnames(task)
    #classNameIndex=0

    '''
    creates a folder for contour images and contour points. Contour images are used for sanity check, and contour points are passed to context generation
    '''
    #for label_class in label_classes:
        #I.findAllContours(label_class,label_classNames[classNameIndex],SAVE_TEST_IMAGE=True,SAVE_DATA=True)
        #I.findAllContoursTimed(label_class,label_classNames[classNameIndex],SAVE_TEST_IMAGE=True,SAVE_DATA=True)
        #classNameIndex+=1

    I = Context_Iterator(TASK,CWD)
    #I.CheckDataIntegrity()
    I.GenerateContext(SAVE=True)
    end_time = time.time()

    I = Metrics_Iterator(TASK,CWD)
    I.IOU()

    print("--- %s seconds ---" % (end_time - start_time))
    
    quit();

'''
def getLabelClassnames(task):
    if "Knot" in task:
        return ["2023_grasper_L_masks","2023_grasper_R_masks","2023_thread_masks"], ["2023_grasper_L","2023_grasper_R","2023_thread" ]
    elif "Needle" in task:
        return ["2023_grasper_L_masks","2023_grasper_R_masks","2023_thread_masks"], ["2023_grasper_L","2023_grasper_R","2023_thread" ] # add Needle,
    elif "Suturing" in task:
        return ["2023_grasper_L_masks","2023_grasper_R_masks","2023_thread_masks"], ["2023_grasper_L","2023_grasper_R","2023_thread" ] # add Needle
'''


main();
