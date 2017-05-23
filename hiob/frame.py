"""
Created on 2016-11-29

@author: Peer Springstübe
"""

import logging

import transitions

logger = logging.getLogger(__name__)


class Frame(object):

    module_states = [
        'created',
        'capturing',
        'captured',
        'calculating_roi',
        'roi_calculated',
        'generating_sroi',
        'sroi_generated',
        'extracting_features',
        'features_extracted',
        'reducing_features',
        'features_reduced',
        'consolidating_features',
        'features_consolidated',
        'pursuing',
        'pursued',
        'evaluating',
        'evaluated',
    ]
    transitions = [
        ['commence_capture', 'created', 'capturing'],
        ['complete_capture', 'capturing', 'captured'],
        ['commence_roi', 'captured', 'calculating_roi'],
        ['complete_roi', 'calculating_roi', 'roi_calculated'],
        ['commence_sroi', 'roi_calculated', 'generating_sroi'],
        ['complete_sroi', 'generating_sroi', 'sroi_generated'],
        ['commence_extraction', 'sroi_generated', 'extracting_features'],
        ['complete_extraction', 'extracting_features', 'features_extracted'],
        ['commence_reduction', 'features_extracted', 'reducing_features'],
        {'trigger': 'complete_reduction', 'source': 'reducing_features',
            'dest': 'features_reduced', 'after': '_after_reduction'},
        ['commence_consolidation', 'features_reduced',
            'consolidating_features'],
        ['complete_consolidation', 'consolidating_features',
            'features_consolidated'],
        ['commence_pursuing', 'features_consolidated', 'pursuing'],
        ['complete_pursuing', 'pursuing', 'pursued'],
        ['commence_evaluation', 'pursued', 'evaluating'],
        ['complete_evaluation', 'evaluating', 'evaluated'],
    ]

    def __init__(self, tracking, number):
        logger.info("Creating new Frame")
        # the tracking process this frame is part of:
        self.tracking = tracking
        # number of this frame within the tracking process:
        self.number = number

        # state machine for this frame:
        self.machine = transitions.Machine(
            model=self,
            states=self.module_states, initial='created',
            transitions=self.transitions,
            name=self.__repr__(),
        )

        self.ground_truth = None
        self.capture_image = None
        self.previous_position = None
        self.roi = None
        self.sroi_image = None
        self.features = None
        self.consolidated_features = None
        self.predicted_position = None
        self.prediction_quality = None
        self.target_mask = None
        self.result = None
        self.lost = 0
        self.updated = 'n'

        # state markers - only to be changed by state machine
        self.did_reduction = False

    def number_string(self):
        if self.tracking and self.tracking.total_frames:
            l = len(str(self.tracking.total_frames))
            fmt = "%0" + str(l) + "d/" + str(self.tracking.total_frames)
            return fmt % self.number
        else:
            return str(self.number)

    def __repr__(self):
        return "<Frame#{} of Tracking#{} on {}>".format(self.number_string(), self.tracking.serial, self.tracking.sample.full_name)

    def _after_reduction(self):
        self.did_reduction = True
