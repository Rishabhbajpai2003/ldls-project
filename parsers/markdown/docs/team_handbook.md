# Platform Team Handbook

Welcome to the Platform team. This handbook explains how we work. It is a living document; open a pull request if something is out of date.

## Communication

We use the channel `#platform-team` for day-to-day questions and `#platform-alerts` for automated alerts only. Stand-up is at 09:45 every weekday and lasts no longer than fifteen minutes.

## On-call

### Rotation

On-call shifts last one week and start every Wednesday at 10:00. Each engineer is on call about once every seven weeks. Swaps are allowed if you update the schedule yourself.

### Responding to pages

Acknowledge a page within 10 minutes. If you cannot fix the problem within 30 minutes, escalate to the secondary on-call engineer. Every page that wakes someone up at night gets a short review the next day.

### Compensation

Each on-call week earns one extra day off, to be taken within the following two months.

## Code review

- Every change needs one approval from a team member.
- Changes to the database schema need two approvals, one of them from the data lead.
- Reviewers should respond within one working day.
- Keep pull requests under 400 changed lines where possible.

## Release process

### Schedule

We release on Tuesdays and Thursdays. No releases on Fridays or the day before a public holiday.

### Steps

1. Merge to `main` and wait for the green build.
2. Tag the commit with the version, for example `v5.12.0`.
3. Deploy to staging and run the smoke tests.
4. Deploy to production in three waves: 5 percent, 50 percent, 100 percent.

### Rollback

If error rates rise above 2 percent during a wave, stop and roll back with `deployctl rollback`. Write a note in `#platform-team` explaining what happened.

## Time off

Request holidays at least two weeks in advance in the HR portal. Tell the team in stand-up and make sure your on-call shift is covered. The team avoids having more than two people away at the same time.

## Learning budget

Every engineer has a yearly learning budget of 1,200 euros for books, courses and conferences. Ask your manager before spending more than 300 euros at once.
