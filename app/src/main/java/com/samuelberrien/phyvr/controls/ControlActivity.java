package com.samuelberrien.phyvr.controls;

import android.os.Bundle;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import com.samuelberrien.phyvr.R;

import java.util.ArrayList;

public class ControlActivity extends AppCompatActivity {

	private ArrayList<SetUpControl> setUpControls;

	@Override
	protected void onCreate(@Nullable Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_control);

		setUpControls = new ArrayList<>();

		setUpControls.add(new SetUpControl(this, R.string.left_control, this));
		setUpControls.add(new SetUpControl(this, R.string.right_control, this));
		setUpControls.add(new SetUpControl(this, R.string.speed_up_control, this));
		setUpControls.add(new SetUpControl(this, R.string.speed_down_control, this));

		LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);

		for (SetUpControl s : setUpControls)
			((LinearLayout) findViewById(R.id.linearlayout_control)).addView(s, params);
	}

	@Override
	public boolean dispatchGenericMotionEvent(MotionEvent e) {
		for (SetUpControl s : setUpControls)
			s.motionEvent(e);
		return true;
	}

	@Override
	public boolean dispatchKeyEvent(KeyEvent event) {
		for (SetUpControl s : setUpControls)
			s.keyDown(event.getKeyCode(), event);
		return event.getKeyCode() != KeyEvent.KEYCODE_BACK || super.dispatchKeyEvent(event);
	}
}
