program reference_label_binarization_driver
  use sklearn_label_binarization, only: label_binarizer,multi_label_binarizer,label_set
  use sklearn_common, only: dp
  implicit none
  type(label_binarizer) :: lb
  type(multi_label_binarizer) :: mlb
  type(label_set), allocatable :: train_sets(:),query_sets(:),restored(:)
  integer, allocatable :: train(:),query(:),z(:,:),q(:,:),decoded(:)
  real(dp), allocatable :: scores(:,:)
  integer :: input,output,kind,n,m,k,i,threshold_set,width
  real(dp) :: threshold
  character(512) :: file
  call get_command_argument(1,file); open(newunit=input,file=trim(file),status='old')
  call get_command_argument(2,file); open(newunit=output,file=trim(file),status='replace')
  read(input,*) kind,n,m
  if(kind==1) then
    read(input,*) lb%neg_label,lb%pos_label,width,threshold_set,threshold
    allocate(train(n),query(m),scores(m,width))
    read(input,*) train; read(input,*) query
    do i=1,m
      read(input,*) scores(i,:)
    end do
    call lb%fit(train); z=lb%transform(train)
    call emit('fit',reshape(z,[size(z)]))
    z=lb%fit_transform(train); q=lb%transform(query)
    call emit('classes',lb%classes_); call emit('train',reshape(z,[size(z)]))
    call emit('query',reshape(q,[size(q)])); call emit('shape',shape(q))
    call emit('type',[merge(1,0,trim(lb%y_type_)=='multiclass')])
    decoded=lb%inverse_transform(q); call emit('inverse',decoded)
    if(threshold_set==1) then
      decoded=lb%inverse_transform(scores,threshold)
    else
      decoded=lb%inverse_transform(scores)
    end if
    call emit('scores',decoded)
    lb%neg_label=-5; lb%pos_label=3; q=lb%transform(query)
    call emit('live',reshape(q,[size(q)]))
    z=lb%fit_transform([99,98,99])
    call emit('refit_classes',lb%classes_); call emit('refit',reshape(z,[size(z)]))
  else
    read(input,*) k
    if(k>=0) then
      allocate(mlb%classes(k))
      if(k>0) read(input,*) mlb%classes
    end if
    allocate(train_sets(n),query_sets(m))
    do i=1,n
      call read_set(train_sets(i))
    end do
    do i=1,m
      call read_set(query_sets(i))
    end do
    call mlb%fit(train_sets); z=mlb%transform(train_sets)
    call emit('fit',reshape(z,[size(z)]))
    z=mlb%fit_transform(train_sets); q=mlb%transform(query_sets)
    call emit('classes',mlb%classes_); call emit('train',reshape(z,[size(z)]))
    call emit('query',reshape(q,[size(q)])); call emit('shape',shape(q))
    restored=mlb%inverse_transform(q); call emit_sets('inverse',restored)
    restored=mlb%inverse_transform(real(1-q,dp)); call emit_sets('complement',restored)
    if(allocated(mlb%classes)) mlb%classes=[integer ::]
    q=mlb%transform(query_sets); call emit('snapshot',reshape(q,[size(q)]))
    if(allocated(mlb%classes)) deallocate(mlb%classes)
    z=mlb%fit_transform(query_sets)
    call emit('refit_classes',mlb%classes_); call emit('refit',reshape(z,[size(z)]))
  end if
  close(input); close(output)
contains
  subroutine emit(name,values)
    character(*), intent(in) :: name
    integer, intent(in) :: values(:)
    write(output,'(a,*(1x,i0))') name,values
  end subroutine
  subroutine read_set(row)
    type(label_set), intent(out) :: row
    integer :: count
    read(input,*) count
    allocate(row%values(count))
    if(count>0) read(input,*) row%values
  end subroutine
  subroutine emit_sets(name,rows)
    character(*), intent(in) :: name
    type(label_set), intent(in) :: rows(:)
    integer, allocatable :: counts(:),values(:)
    integer :: i,first,n
    allocate(counts(size(rows)))
    do i=1,size(rows)
      counts(i)=size(rows(i)%values)
    end do
    allocate(values(sum(counts))); first=1
    do i=1,size(rows)
      n=counts(i); values(first:first+n-1)=rows(i)%values; first=first+n
    end do
    call emit(name//'_counts',counts); call emit(name//'_values',values)
  end subroutine
end program
