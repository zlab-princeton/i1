from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
pipeline(Tasks.visual_question_answering, model='damo/mplug_visual-question-answering_coco_large_en', device='cpu')