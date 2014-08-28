IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;

//This fucntion recives a label dataset in format NumericField
//and convertsit to matrix format, each column is coresponded to one sample
//in each column the element correponded to the class label would be 1, all other elements would be zero
// a label dataset if like [{1,1,1}, {2,1,2}] would be converted to [{1,1,1},{2,1,0},{1,2,0},{2,2,1}]
EXPORT ToGroundTruth(DATASET(ML.Types.NumericField) y ) := FUNCTION


zero_mat    := DATASET ([{1,1,0}],$.M_Types.MatRecord);
sample_num  := MAX (y,y.id);
class_num   := MAX (y, y.value);
scratch_mat := ML.Mat.Repmat (zero_mat, class_num, sample_num);



$.M_Types.MatRecord ToGT(scratch_mat l, y r) := TRANSFORM
  SELF.value := IF(l.x=r.value,1,0);
  SELF := l;
END;


Result := JOIN (scratch_mat, y, LEFT.y=RIGHT.id , ToGT(LEFT,RIGHT));

RETURN Result;

END;