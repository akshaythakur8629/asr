import tempfile, unittest
from pathlib import Path
from utils.diarize_inventory import parse_rttm
class DiarizationTests(unittest.TestCase):
    def test_parse_sort_and_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'x.rttm';path.write_text('SPEAKER x 1 1.000 1.000 <NA> <NA> speaker_1 <NA> <NA>\nSPEAKER x 1 0.000 1.500 <NA> <NA> speaker_0 <NA> <NA>\n')
            turns=parse_rttm(path);self.assertEqual([x.speaker for x in turns],['speaker_0','speaker_1']);self.assertTrue(all(x.overlap_flag for x in turns))
if __name__=='__main__':unittest.main()
