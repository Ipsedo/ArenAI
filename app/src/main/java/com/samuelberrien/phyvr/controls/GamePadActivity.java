package com.samuelberrien.phyvr.controls;

import android.os.Bundle;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
import android.widget.LinearLayout;
import com.samuelberrien.phyvr.R;

public class GamePadActivity extends AppCompatActivity {

	@Override
	protected void onCreate(@Nullable Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_gamepad);
	}

	@Override
	public boolean dispatchGenericMotionEvent(MotionEvent e) {
		boolean handled = false;
		LinearLayout axises = findViewById(R.id.axis_set_up_layout);
		for (int i = 0; i < axises.getChildCount(); i++) {
			handled = axises.getChildAt(i).onGenericMotionEvent(e) || handled;
		}
		return handled || super.dispatchGenericMotionEvent(e);
	}

	@Override
	public boolean dispatchKeyEvent(KeyEvent event) {
		boolean handled = false;
		LinearLayout buttons = findViewById(R.id.button_set_up_layout);
		for (int i = 0; i < buttons.getChildCount(); i++) {
			View v = buttons.getChildAt(i);
			if (event.getAction() == KeyEvent.ACTION_DOWN) {
				handled = v.onKeyDown(event.getKeyCode(), event) || handled;
			} else if (event.getAction() == KeyEvent.ACTION_UP) {
				handled = v.onKeyUp(event.getKeyCode(), event) || handled;
			}
		}
		return handled || super.dispatchKeyEvent(event);
	}
}
