package com.samuelberrien.phyvr.controls.axis;

import android.content.Context;
import android.support.v4.content.ContextCompat;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.LinearLayout;
import android.content.SharedPreferences;
import android.widget.RelativeLayout;
import android.widget.TextView;
import com.samuelberrien.phyvr.R;
import com.samuelberrien.phyvr.controls.Controls;

import java.util.Timer;
import java.util.TimerTask;

public class AxisContener extends RelativeLayout implements SharedPreferences.OnSharedPreferenceChangeListener, View.OnClickListener {

	private Axis axis;
	private AxisGage axisGage;
	private TextView text;
	private boolean isPlus;
	private boolean hasClick;
	private boolean listening;
	private SharedPreferences pref;

	public AxisContener(Context context, Axis axis, boolean isPlus) {
		super(context);

		this.axis = axis;
		this.isPlus = isPlus;
		hasClick = false;
		listening = false;

		setBackground(ContextCompat.getDrawable(context, R.drawable.button_axis));

		axisGage = new AxisGage(context, axis, isPlus);

		pref = context.getSharedPreferences(Axis.axisPref, Context.MODE_PRIVATE);
		pref.registerOnSharedPreferenceChangeListener(this);

		text = new TextView(context);
		text.setGravity(Gravity.CENTER);
		text.setBackground(ContextCompat.getDrawable(context, android.R.color.transparent));
		text.setText(pref.getInt(axis.getAxisMap().getName() + (isPlus ? "+" : "-"), -1) + (isPlus ? "+" : "-"));

		addView(axisGage, axisGage.makeLayoutParams());
		RelativeLayout.LayoutParams params = new RelativeLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT);
		params.addRule(RelativeLayout.CENTER_IN_PARENT);
		int px = (int) TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, 4,
				getContext().getResources().getDisplayMetrics());
		params.setMargins(px, px, px, px);
		addView(text, params);

		text.setOnClickListener(this);
	}

	@Override
	public void onSharedPreferenceChanged(SharedPreferences pref, String key) {
		if (key.contains(axis.getAxisMap().getName()) && key.contains(isPlus ? "+" : "-")) {
			text.setText(pref.getInt(key, -1) + (isPlus ? "+" : "-"));
		}
	}

	@Override
	public boolean onGenericMotionEvent(MotionEvent event) {
		if (!listening)
			return super.onGenericMotionEvent(event);

		System.out.println("TTTTTTTT");
		for (int i = MotionEvent.AXIS_X; i < MotionEvent.AXIS_GENERIC_16; i++) {
			float v = event.getAxisValue(i);
			if (v > Axis.LIMIT || v < -Axis.LIMIT) {
				pref.edit().putInt(axis.getAxisMap().getName() + (isPlus ? "+" : "-"), i).apply();
				text.setBackground(ContextCompat.getDrawable(getContext(), android.R.color.transparent));
				text.requestLayout();
				listening = false;
			}
		}
		return super.onGenericMotionEvent(event);
	}

	@Override
	public void onClick(View v) {
		if (hasClick && !listening) {
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
			hasClick = false;
			return;
		} else if (hasClick)
			hasClick = false;
		hasClick = true;
	}
}
