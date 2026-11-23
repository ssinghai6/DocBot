# rails runner: upsert wiki pages from /tmp/wiki.json ([{title, slug, parent, text}]), keyed by slug.
p = Project.find_by!(identifier: "docbot")
User.current = User.find_by!(login: "admin")
w = p.wiki
pages = JSON.parse(File.read("/tmp/wiki.json"))
by_slug = {}
pages.each do |d|
  pg = w.pages.find_by(slug: d["slug"]) || w.pages.find_by(title: d["title"]) || WikiPage.new(wiki: w)
  pg.title = d["title"]
  pg.author ||= User.current
  pg.text = d["text"]
  pg.save! if pg.new_record?
  (d["attachments"] || []).each do |name|
    a = pg.attachments.find_by(filename: name) || begin
      mime = name.end_with?(".png") ? "image/png" : "text/html"
      at = Attachment.new(container: pg, author: User.current, description: "")
      at.file = Rack::Multipart::UploadedFile.new("/tmp/wikiatt/#{name}", mime)
      at.save!; at
    end
    pg.text = pg.text.gsub("ATTACH:#{name}", "/api/v3/attachments/#{a.id}/content")
  end
  pg.journal_notes = "import from repo docs" if pg.respond_to?(:journal_notes=)
  pg.save!
  pg.update_column(:slug, d["slug"]) if pg.slug != d["slug"]
  by_slug[d["slug"]] = pg
  puts "#{pg.previous_changes.empty? ? 'ok ' : 'set'} #{d['slug']}"
end
pages.each do |d|
  par = d["parent"] && by_slug[d["parent"]]
  pg = by_slug[d["slug"]]
  pg.update_column(:parent_id, par&.id) if pg.parent_id != par&.id
end
home = by_slug["home"]
w.update!(start_page: home.title) if home && w.start_page != home.title
puts "pages=#{w.pages.count} start=#{w.reload.start_page}"
