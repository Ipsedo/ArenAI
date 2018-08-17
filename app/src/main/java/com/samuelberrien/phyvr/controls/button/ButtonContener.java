package com.samuelberrien.phyvr.controls.button;

import android.content.Context;
import android.support.v4.content.ContextCompat;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.KeyEvent;
import android.view.View;
import android.widget.RelativeLayout;
import android.content.SharedPreferences;
import android.widget.TextView;
import com.samuelberrien.phyvr.R;

import java.security.Key;
import java.util.Timer;
import java.util.TimerTask;

public class ButtonContener extends RelativeLayout implements SharedPreferences.OnSharedPreferenceChangeListener, View.OnClickListener {

	private TextView text;
	private ButtonWitness witness;
	private Button button;
	private boolean listening;
	private SharedPreferences pref;

	public ButtonContener(Context context, Button button) {
		super(context);

		listening = false;
		this.button = button;

		setBackground(ContextCompat.getDrawable(context, R.drawable.button_axis));

		RelativeLayout.LayoutParams layoutParams = new RelativeLayout.LayoutParams(
				RelativeLayout.LayoutParams.MATCH_PARENT, RelativeLayout.LayoutParams.MATCH_PARENT);
		int px = (int) TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 4,
				getContext().getResources().getDisplayMetrics());
		layoutParams.setMargins(px, px, px, px);
		layoutParams.addRule(RelativeLayout.CENTER_VERTICAL);

		text = new TextView(context);
		text.setGravity(Gravity.CENTER);
		text.setBackground(ContextCompat.getDrawable(context, android.R.color.transparent));

		pref = context.getSharedPreferences(Button.buttonPref, Context.MODE_PRIVATE);
		setText(pref);
		pref.registerOnSharedPreferenceChangeListener(this);

		witness = new ButtonWitness(context, button);

		addView(witness, layoutParams);
		addView(text, layoutParams);

		text.setOnClickListener(this);
	}

	private void setText(SharedPreferences pref) {
		int id = pref.getInt(button.getMap().getName(), -1);
		text.setText(Integer.toString(id));
	}

	@Override
	public boolean onKeyDown(int keyCode, KeyEvent event) {
		if (!listening) {
			button.onKeyDown(event);
			return super.onKeyDown(keyCode, event);
		}

		boolean handled = false;
		for (int i = KeyEvent.KEYCODE_BUTTON_A; i < KeyEvent.KEYCODE_BUTTON_MODE; i++) {
			if (keyCode == i) {
				pref.edit().putInt(button.getMap().getName(), keyCode).apply();
				text.setBackground(ContextCompat.getDrawable(getContext(), android.R.color.transparent));
				text.requestLayout();
				listening = false;
				handled = true;
				break;
			}
		}
		return handled || super.onKeyDown(keyCode, event);
	}

	@Override
	public boolean onKeyUp(int keyCode, KeyEvent event) {
		if (!listening)
			button.onKeyUp(event);
		return super.onKeyDown(keyCode, event);
	}

	@Override
	public void onClick(View v) {
		if (!listening) {
			text.setBackground(ContextCompat.getDrawable(getContext(), R.color.redTransparent));
			text.requestLayout();
			new Timer().schedule(new TimerTask() {
				@Override
				public void run() {
					post(() -> {
						text.setBackground(ContextCompat.getDrawable(getContext(), android.R.color.transparent));
						text.requestLayout();
						listening = false;
					});
				}
			}, 4000);
			listening = true;
		}
	}

	@Override
	public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String s) {
		if (s.contains(button.getMap().getName()))
			setText(sharedPreferences);
	}
}
