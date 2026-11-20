program reference_bicluster_driver
  use sklearn_bicluster, only: spectral_coclustering,spectral_biclustering
  use sklearn_common, only: dp
  implicit none
  type(spectral_coclustering) :: co
  type(spectral_biclustering) :: bi
  real(dp), allocatable :: x(:,:),values(:,:)
  integer, allocatable :: ri(:),ci(:)
  logical, allocatable :: rows(:,:),columns(:,:)
  integer :: kind,n,p,input,output,i,j,k,n_init,seed,kr,kc,components,best
  logical :: mini
  character(16) :: method,init
  character(512) :: path
  character(32) :: prefix
  call get_command_argument(1,path); open(newunit=input,file=trim(path),status='old')
  read(input,*) kind,n,p,kr,kc,components,best,n_init,seed,mini
  read(input,*) method,init
  allocate(x(n,p))
  do i=1,n
    read(input,*) x(i,:)
  end do
  close(input)
  call get_command_argument(2,path); open(newunit=output,file=trim(path),status='replace')
  if(kind==0) then
    co%n_clusters=kr; co%n_init=n_init; co%random_state=seed; co%mini_batch=mini; co%init=init
    call co%fit(x)
    call common(co%row_labels_,co%column_labels_,co%rows_,co%columns_,co%normalized_data_,co%singular_values_, &
      co%left_vectors_,co%right_vectors_,co%n_normalization_iter_,co%normalization_converged_)
    call vector('row_scale',co%row_scaling_); call vector('column_scale',co%column_scaling_)
    call matrix('embedding',co%embedding_); call vector('cost',[co%inertia_])
    call vector('meta',real([co%n_rows_,co%n_features_in_,co%n_biclusters_],dp))
    call co%get_biclusters(rows,columns)
    k=co%n_biclusters_
  else
    bi%n_row_clusters=kr; bi%n_column_clusters=kc; bi%n_components=components; bi%n_best=best
    bi%n_init=n_init; bi%random_state=seed; bi%mini_batch=mini; bi%method=method; bi%init=init
    call bi%fit(x)
    call common(bi%row_labels_,bi%column_labels_,bi%rows_,bi%columns_,bi%normalized_data_,bi%singular_values_, &
      bi%left_vectors_,bi%right_vectors_,bi%n_normalization_iter_,bi%normalization_converged_)
    call vector('left_errors',bi%left_piecewise_errors_); call vector('right_errors',bi%right_piecewise_errors_)
    call vector('best_left',real(bi%best_left_indices_,dp)); call vector('best_right',real(bi%best_right_indices_,dp))
    call matrix('row_projection',bi%row_projection_); call matrix('column_projection',bi%column_projection_)
    call vector('meta',real([bi%n_rows_,bi%n_features_in_,bi%n_biclusters_],dp))
    call bi%get_biclusters(rows,columns)
    k=bi%n_biclusters_
  end if
  call matrix('copied_rows',real(merge(1,0,rows),dp)); call matrix('copied_columns',real(merge(1,0,columns),dp))
  do i=0,k-1
    write(prefix,'(a,i0,a)') 'block',i,'_'
    if(kind==0) then
      call co%get_indices(i,ri,ci); call vector(trim(prefix)//'shape',real(co%get_shape(i),dp))
      values=co%get_submatrix(i,x)
    else
      call bi%get_indices(i,ri,ci); call vector(trim(prefix)//'shape',real(bi%get_shape(i),dp))
      values=bi%get_submatrix(i,x)
    end if
    call vector(trim(prefix)//'rows',real(ri,dp)); call vector(trim(prefix)//'columns',real(ci,dp))
    call matrix(trim(prefix)//'data',values)
  end do
  close(output)
contains
  subroutine common(rl,cl,rows,columns,normalized,s,left,right,norm_iter,converged)
    integer, intent(in) :: rl(:),cl(:),norm_iter
    logical, intent(in) :: rows(:,:),columns(:,:),converged
    real(dp), intent(in) :: normalized(:,:),s(:),left(:,:),right(:,:)
    call vector('row_labels',real(rl,dp)); call vector('column_labels',real(cl,dp))
    call matrix('rows',real(merge(1,0,rows),dp)); call matrix('columns',real(merge(1,0,columns),dp))
    call matrix('normalized',normalized); call vector('singular',s)
    call matrix('left',left); call matrix('right',right)
    call vector('normalization',[real(norm_iter,dp),real(merge(1,0,converged),dp)])
  end subroutine
  subroutine matrix(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:,:)
    call vector(name,reshape(a,[size(a)]))
  end subroutine
  subroutine vector(name,a)
    character(*), intent(in) :: name
    real(dp), intent(in) :: a(:)
    write(output,'(a,1x,*(es25.16e3,1x))') name,a
  end subroutine
end program
