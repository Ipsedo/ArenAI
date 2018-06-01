package com.samuelberrien.phyvr.controls;

import android.app.Activity;

public class SetUpControlButton extends SetUpControl {
	public SetUpControlButton(Activity activity, int controlStrId) {
		super(activity, controlStrId);
		infos.isMotionEvent = false;
	}
}
