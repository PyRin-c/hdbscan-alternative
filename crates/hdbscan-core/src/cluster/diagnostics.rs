use std::cell::RefCell;

#[derive(Clone, Copy, Debug)]
pub enum SelectionDiagnostic {
    BranchGraphExtrema { core: bool },
    LambdaCast([f64; 2]),
    InvalidLifetime,
    ThresholdCastOverflow,
    BranchLambdaReciprocal(f64),
    RootLambdaReciprocal([f64; 3]),
}

thread_local! {
    static DIAGNOSTICS: RefCell<Option<Vec<SelectionDiagnostic>>> = const { RefCell::new(None) };
}

struct Capture(Option<Vec<SelectionDiagnostic>>);

impl Drop for Capture {
    fn drop(&mut self) {
        DIAGNOSTICS.with(|events| events.replace(self.0.take()));
    }
}

/// Collects diagnostics emitted while this thread runs the fit, including on error.
pub fn capture_selection_diagnostics<T>(fit: impl FnOnce() -> T) -> (T, Vec<SelectionDiagnostic>) {
    let _capture = Capture(DIAGNOSTICS.with(|events| events.replace(Some(Vec::new()))));
    let result = fit();
    let events = DIAGNOSTICS.with(|events| events.take().unwrap_or_default());
    (result, events)
}

pub(crate) fn report(event: SelectionDiagnostic) {
    DIAGNOSTICS.with_borrow_mut(|events| {
        if let Some(events) = events {
            events.push(event);
        }
    });
}
