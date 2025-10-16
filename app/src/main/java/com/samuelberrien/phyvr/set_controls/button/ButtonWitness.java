package com.samuelberrien.phyvr.set_controls.button;

import android.content.Context;
import androidx.core.content.ContextCompat;
import android.view.View;
import com.samuelberrien.phyvr.R;

public class ButtonWitness extends View implements Button.OnStateChangedListener {

	public ButtonWitness(Context context, Button button) {
		super(context);
		button.addListener(this);
		setBackground(ContextCompat.getDrawable(context, android.R.color.transparent));
	}

	@Override
	public void onStateChanged(boolean newState) {
		if (newState)
			setBackground(ContextCompat.getDrawable(getContext(), R.color.colorAccent));
		else
			setBackground(ContextCompat.getDrawable(getContext(), android.R.color.transparent));
		requestLayout();
	}
}
