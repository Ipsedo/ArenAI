package com.samuelberrien.phyvr.utils;

import android.content.Context;
import android.support.v4.content.ContextCompat;
import android.util.TypedValue;

public class Dimens {
	public static int dpResToPx(Context context, int dpResId) {
		float dp = context.getResources().getDimension(dpResId) / context.getResources().getDisplayMetrics().density;
		return  (int) TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, dp,
				context.getResources().getDisplayMetrics());
	}
}
