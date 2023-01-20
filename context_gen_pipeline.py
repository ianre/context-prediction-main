import os, sys
from pipeline_scripts.mask_context_evaluation import Context_Iterator
from pipeline_scripts.contour_extraction import Contour_Iterator
from pipeline_scripts.metrics import Metrics_Iterator
import pipeline_scripts.utils

def main():
    CWD=os.getcwd()
    task = "Knot_Tying"
    try:
        task=sys.argv[1]
    except:
        print("Error: no task provided","Usage: python context_gen_pipeline.py <task>","Default Task"+task)

    print("Pipeline Running for task ",task)

    print("Creating contour extractor")
    I = Contour_Iterator(task,CWD)
    label_classes = ["2023_grasper_L_masks","2023_grasper_R_masks","2023_thread_masks"] #,"deeplab_needle_v3"
    label_classNames = ["2023_grasper_L","2023_grasper_R","2023_thread" ] # ,"dl_needle"


    classNameIndex=0

    for label_class in label_classes:
        '''
        creates a folder for contour images and contour points. Contour images are used for sanity check, and contour points are passed to context generation
        '''
        #I.findAllContours(label_class,label_classNames[classNameIndex],SAVE_TEST_IMAGE=True,SAVE_DATA=True)
        classNameIndex+=1



    I = Context_Iterator(task,CWD)
    #I.CheckDataIntegrity()
    I.GenerateContext(SAVE=True)



    I = Metrics_Iterator(task,CWD)
    I.IOU()
    
    quit();

main();
