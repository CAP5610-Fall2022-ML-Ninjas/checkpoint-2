const gulp = require('gulp');
const connect = require('gulp-connect');

const sources = ['*.html', '*.md'];

reload = () => gulp.src(sources).pipe(connect.reload());

exports.default = (taskComplete) => {
    gulp.watch(sources, reload)
    connect.server({
        root: '.',
        port: 8000,
        host: 'localhost',
        livereload: true
    })
    taskComplete()
}
