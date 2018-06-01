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

		setUpControls.add(new SetUpControlAxis(this, R.string.left_control));
		setUpControls.add(new SetUpControlAxis(this, R.string.right_control));
		setUpControls.add(new SetUpControlAxis(this, R.string.speed_up_control));
		setUpControls.add(new SetUpControlAxis(this, R.string.speed_down_control));
		setUpControls.add(new SetUpControlButton(this, R.string.brake_control));
		setUpControls.add(new SetUpControlAxis(this, R.string.turret_left_control));
		setUpControls.add(new SetUpControlAxis(this, R.string.turret_right_control));
		setUpControls.add(new SetUpControlAxis(this, R.string.turret_up_control));
		setUpControls.add(new SetUpControlAxis(this, R.string.turret_down_control));
		setUpControls.add(new SetUpControlButton(this, R.string.respawn_control));
		setUpControls.add(new SetUpControlButton(this, R.string.fire_control));

		LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);

		for (SetUpControl s : setUpControls)
			((LinearLayout) findViewById(R.id.linearlayout_control)).addView(s, params);
	}

	@Override
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
	}

	@Override
	protected void onPause() {
		for (SetUpControl s : setUpControls)
			s.onPause();
		super.onPause();
	}

	@Override
	protected void onResume() {
		super.onResume();
		for (SetUpControl s : setUpControls)
			s.onResume();
	}
}
