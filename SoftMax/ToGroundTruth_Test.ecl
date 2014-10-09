IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;

//input data
value_record := RECORD
			unsigned      id;
      real      f1;
			real      f2;
			real      f3;
			real      f4;
      integer1        label; 
END;
input_data := DATASET([
{1, 0.1, 0.2, 0.5 , 0.7, 1},
{2, 0.8, 0.9, 0.4, 0.9 , 2},
{3, 0.5, 0.9, 0.1, 0.1, 3},
{4, 0.8, 0.7, 0.4, 0.6, 3},
{5, 0.9,0.1, 0.3, 0.2, 2},
{6, 0.1, 0.3, 0.6, 0.8, 1}],
 value_record);
OUTPUT  (input_data, ALL, NAMED ('input_data'));
//convert input data to two matrices, samples matrix and labels matrix
Sampledata_Format := RECORD 
			input_data.id;
			input_data.f1;
			input_data.f2;
			input_data.f3;
			input_data.f4;

END;

sample_table := TABLE(input_data,Sampledata_Format);

labeldata_Format := RECORD 
			input_data.id;
			input_data.label;

END;

label_table := TABLE(input_data,labeldata_Format);
OUTPUT  (label_table, ALL, NAMED ('label_table'));


//convert sample_table to matrix format

ML.ToField(sample_table, sample_table_in_numeric_field_format)
OUTPUT  (sample_table, ALL, NAMED ('sample_table'));
OUTPUT  (sample_table_in_numeric_field_format, ALL, NAMED ('sample_table_in_numeric_field_format'));

ML.ToField(label_table, label_table_in_numeric_field_format)
OUTPUT  (label_table_in_numeric_field_format, ALL, NAMED ('label_table_in_numeric_field_format'));


 y := label_table_in_numeric_field_format;


zero_mat    := DATASET ([{1,1,0}],$.M_Types.MatRecord);
sample_num  := MAX (y,y.id);
class_num   := MAX (y, y.value);
scratch_mat := ML.Mat.Repmat (zero_mat, class_num, sample_num);
OUTPUT  (scratch_mat, ALL, NAMED ('scratch_mat'));


$.M_Types.MatRecord ToGT(scratch_mat l, y r) := TRANSFORM
  SELF.value := IF(l.x=r.value,1,0);
  SELF := l;
END;


Result := JOIN (scratch_mat, y, LEFT.y=RIGHT.id , ToGT(LEFT,RIGHT));

OUTPUT  (Result, ALL, NAMED ('Result'));


result2 := $.ToGroundTruth (y);
OUTPUT  (result2, ALL, NAMED ('result2'));