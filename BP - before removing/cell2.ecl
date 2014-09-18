IMPORT * FROM ML;
IMPORT ML.Mat;
IMPORT $;

EXPORT Cell2(DATASET($.M_Types.CellMatRec) w, DATASET($.M_Types.CellMatRec) b) := FUNCTION

$.M_Types.CellMatRec2 J2 (w l,b r) := TRANSFORM 
	SELF.cellmat1 := l.cellmat;
	SELF.cellmat2 := r.cellmat;
	SELF 					:= l;
END;

wb := JOIN(w,b,(LEFT.id=RIGHT.id),J2(LEFT,RIGHT)); 
RETURN wb;
END;