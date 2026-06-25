import tempfile, unittest
from pathlib import Path
from utils.pipeline import JobStore, _merge_and_label_turns
class PipelineTests(unittest.TestCase):
    def test_audio_path_rejects_unknown_kind(self):
        with tempfile.TemporaryDirectory() as d:
            store=JobStore(Path(d));self.assertIsNone(store.audio_path('missing','input'))
    def test_merge_same_speaker_drop_empty_and_label_roles(self):
        turns=[
            {'speaker':'speaker_0','start_sec':0.0,'end_sec':1.0,'overlap_flag':False,'text':'hello','canonical_text':'hello','display_text':'hello','spans':[],'itn_deferred':False},
            {'speaker':'speaker_0','start_sec':1.1,'end_sec':2.0,'overlap_flag':False,'text':'ई एम आई','canonical_text':'EMI','display_text':'EMI','spans':[{'start':0,'end':6}],'itn_deferred':False},
            {'speaker':'speaker_1','start_sec':2.0,'end_sec':2.2,'overlap_flag':False,'text':'  ','canonical_text':'  ','display_text':'  ','spans':[],'itn_deferred':False},
            {'speaker':'speaker_1','start_sec':2.3,'end_sec':3.0,'overlap_flag':False,'text':'reply','canonical_text':'reply','display_text':'reply','spans':[],'itn_deferred':False},
        ]
        result=_merge_and_label_turns(turns)
        self.assertEqual([x['speaker'] for x in result],['customer','agent'])
        self.assertEqual(result[0]['text'],'hello ई एम आई')
        self.assertEqual(result[0]['canonical_text'],'hello EMI')
        self.assertEqual(result[0]['spans'][0]['start'],6)
        self.assertEqual(result[0]['spans'][0]['end'],12)
if __name__=='__main__':unittest.main()
