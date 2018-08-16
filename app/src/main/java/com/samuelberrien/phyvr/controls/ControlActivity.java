package com.samuelberrien.phyvr.controls;

import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import com.samuelberrien.phyvr.R;
import com.samuelberrien.phyvr.controls.axis.Axis;

import java.util.ArrayList;

public class ControlActivity extends AppCompatActivity {

	private ArrayList<SetUpControl> setUpControls;

	@Override
	protected void onCreate(@Nullable Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_control);
	}

	/*@Override
	public boolean dispatchGenericMotionEvent(MotionEvent e) {
		boolean handled = false;
		for (SetUpControl s : setUpControls)
			handled = s.motionEvent(e) || handled;
		return handled || super.dispatchGenericMotionEvent(e);
	}

	@Override
	public boolean dispatchKeyEvent(KeyEvent event) {
		boolean handled = false;
		for (SetUpControl s : setUpControls)
			handled = s.keyDown(event.getKeyCode(), event) || handled;
		return handled || super.dispatchKeyEvent(event);
	}*/

	@Override
	protected void onPause() {
		/*for (SetUpControl s : setUpControls)
			s.onPause();*/
		super.onPause();
	}

	@Override
	protected void onResume() {
		super.onResume();
		/*for (SetUpControl s : setUpControls)
			s.onResume();*/
	}
}
