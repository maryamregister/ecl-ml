IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;
//same as ML.Types.FromMatrix
EXPORT MatrixToField(DATASET(ML.Mat.Types.Element) d):=FUNCTION
  RETURN PROJECT(d,TRANSFORM(ML.Types.NumericField,SELF.id:=(TYPEOF(ML.Types.NumericField.id))LEFT.x;SELF.number:=(TYPEOF(ML.Types.NumericField.number))LEFT.y;SELF.value:=(TYPEOF(ML.Types.NumericField.value))LEFT.value;));
END;