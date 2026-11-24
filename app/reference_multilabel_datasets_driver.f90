program reference_multilabel_datasets_driver
  use sklearn_common, only: dp,require
  use sklearn_datasets, only: make_multilabel_classification
  use sklearn_sparse, only: csr_matrix
  use sklearn_label_binarization, only: label_set
  implicit none
  integer :: cases,c,mode,n,p,nc,nl,len,allow,seed,i,j
  real(dp), allocatable :: x(:,:),other(:,:),pc(:),pwc(:,:),values(:)
  integer, allocatable :: y(:,:),other_y(:,:),set_y(:,:),offsets(:),flat(:)
  type(csr_matrix) :: sx,sx2,sy,sy2
  type(label_set), allocatable :: sets(:),sets2(:)
  real(dp) :: consistency(5)
  character(64) :: invalid
  call get_command_argument(1,invalid)
  if(len_trim(invalid)>0) then
    call reject(trim(invalid));error stop 'invalid multilabel generator input accepted'
  end if
  read(*,*) cases
  do c=1,cases
    read(*,*) mode,n,p,nc,nl,len,allow,seed
    write(*,'(a,1x,i0)') 'BEGIN',c
    if(mode==0) then
      call make_multilabel_classification(n,p,x,y,nc,nl,len,allow==1,seed,pc,pwc)
      call make_multilabel_classification(n,p,sx,other_y,nc,nl,len,allow==1,seed)
      consistency(1)=merge(1._dp,0._dp,all(sx%to_dense()==x).and.all(other_y==y))
      call make_multilabel_classification(n,p,other,sy,nc,nl,len,allow==1,seed)
      consistency(2)=merge(1._dp,0._dp,all(other==x).and.all(sy%to_dense()==real(y,dp)))
      call make_multilabel_classification(n,p,sx2,sy2,nc,nl,len,allow==1,seed)
      consistency(3)=merge(1._dp,0._dp,csr_equal(sx,sx2).and.csr_equal(sy,sy2))
      call make_multilabel_classification(n,p,other,sets,nc,nl,len,allow==1,seed)
      set_y=set_indicator(sets,nc)
      consistency(4)=merge(1._dp,0._dp,all(other==x).and.all(set_y==y))
      call make_multilabel_classification(n,p,sx2,sets2,nc,nl,len,allow==1,seed)
      set_y=set_indicator(sets2,nc)
      consistency(5)=merge(1._dp,0._dp,csr_equal(sx,sx2).and.all(set_y==y))
      call matrix('x',x);call matrix('y',real(y,dp));call vector('consistency',consistency)
      call vector('p_c',pc);call matrix('p_w_c',pwc)
      call ragged(sets)
    else
      call make_multilabel_classification(n,p,sx,sy,nc,nl,len,allow==1,seed,pc,pwc)
      call vector('p_c',pc);call vector('word_probability_sums',sum(pwc,dim=1))
    end if
    call sparse_fields('x',sx);call sparse_fields('y',sy)
    call vector('metadata',real([sx%n_rows,sx%n_cols,sy%n_cols],dp))
    write(*,'(a)') 'END';flush(6)
  end do
contains
  logical function csr_equal(a,b) result(ok)
    type(csr_matrix), intent(in) :: a,b
    ok=a%n_rows==b%n_rows.and.a%n_cols==b%n_cols.and.a%nnz()==b%nnz()
    if(.not.ok) return
    ok=all(a%indptr==b%indptr).and.all(a%indices==b%indices).and.all(a%data==b%data)
  end function
  function set_indicator(s,nclasses) result(out)
    type(label_set), intent(in) :: s(:)
    integer, intent(in) :: nclasses
    integer, allocatable :: out(:,:)
    integer :: row
    allocate(out(size(s),nclasses));out=0
    do row=1,size(s)
      out(row,s(row)%values+1)=1
    end do
  end function
  subroutine ragged(s)
    type(label_set), intent(in) :: s(:)
    integer :: row,count_values,pos
    count_values=0
    do row=1,size(s)
      count_values=count_values+size(s(row)%values)
    end do
    allocate(offsets(size(s)+1),flat(count_values));pos=1;offsets(1)=pos
    do row=1,size(s)
      count_values=size(s(row)%values);flat(pos:pos+count_values-1)=s(row)%values
      pos=pos+count_values;offsets(row+1)=pos
    end do
    call vector('set_offsets',real(offsets,dp));call vector('set_labels',real(flat,dp))
    deallocate(offsets,flat)
  end subroutine
  subroutine sparse_fields(prefix,a)
    character(*), intent(in) :: prefix
    type(csr_matrix), intent(in) :: a
    call vector(prefix//'_data',a%data);call vector(prefix//'_indices',real(a%indices,dp))
    call vector(prefix//'_indptr',real(a%indptr,dp))
  end subroutine
  subroutine vector(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:)
    integer :: t
    write(*,'(a,1x,i0)',advance='no') name,size(a)
    do t=1,size(a)
      write(*,'(1x,es25.17e3)',advance='no') a(t)
    end do
    write(*,*)
  end subroutine
  subroutine matrix(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:,:)
    call vector(name,reshape(transpose(a),[size(a)]))
  end subroutine
  subroutine reject(name)
    character(*), intent(in) :: name
    select case(name)
    case('samples');call make_multilabel_classification(0,3,x,y)
    case('rows_limit');call make_multilabel_classification(huge(1),3,sx,sy)
    case('features');call make_multilabel_classification(2,0,x,y)
    case('classes');call make_multilabel_classification(2,3,x,y,n_classes=0)
    case('labels');call make_multilabel_classification(2,3,x,y,n_labels=-1)
    case('length');call make_multilabel_classification(2,3,x,y,length=0)
    case('seed');call make_multilabel_classification(2,3,x,y,random_state=-1)
    case('impossible');call make_multilabel_classification(2,3,x,y,n_labels=0,allow_unlabeled=.false.)
    case default;error stop 'unknown rejection'
    end select
  end subroutine
end program
